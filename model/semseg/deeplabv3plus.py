import model.backbone.resnet as resnet
from model.backbone.xception import xception

import torch
from torch import nn
import torch.nn.functional as F

from model.fgsr_head import FGSR
from model.semseg.resnet_maa_decoder import ResNetMAADecoder
from model.semseg.aspp_maa_hybrid import ASPPMAADecoder


class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3Plus, self).__init__()

        if 'resnet' in cfg['backbone']:
            self.backbone = resnet.__dict__[cfg['backbone']](pretrained=True,
                                                             replace_stride_with_dilation=cfg['replace_stride_with_dilation'])
        else:
            assert cfg['backbone'] == 'xception'
            self.backbone = xception(pretrained=True)

        self.use_maa_decoder = cfg.get('use_maa_decoder', False)
        self.use_hybrid_decoder = cfg.get('use_hybrid_decoder', False)
        # Diagnostic option for the FGSR detach experiment. Default False keeps
        # the original FGSR behaviour unchanged.
        self.detach_fgsr_features = cfg.get('detach_fgsr_features', False)

        if self.use_hybrid_decoder:
            # ASPP-MAA Hybrid Decoder (ASPP primary + MAA residual)
            self.hybrid_decoder = ASPPMAADecoder(num_classes=cfg['nclass'], maa_dim=128)
        elif self.use_maa_decoder:
            # ResNet-adapted MAA Decoder (gated multi-level fusion)
            maa_dim = cfg.get('maa_embed_dim', 128)
            self.maa_decoder = ResNetMAADecoder(num_classes=cfg['nclass'], embed_dim=maa_dim)
        else:
            # Original DeepLabV3+ decoder (ASPP + fuse)
            low_channels = 256
            high_channels = 2048
            self.head = ASPPModule(high_channels, cfg['dilations'])
            self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                        nn.BatchNorm2d(48),
                                        nn.ReLU(True))
            self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(True),
                                      nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(True))
            self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)

        # FGSR: Feature-Guided Spatial Reweighting
        # Only created when use_fgsr=True, avoids checkpoint compatibility issues
        if cfg.get('use_fgsr', False):
            r_min = cfg.get('r_min', 0.5)
            self.fgsr_head = FGSR(embed_dim=128, r_min=r_min)

    def forward(self, x, need_fp=False, return_reliability=False):
        h, w = x.shape[-2:]

        feats = self.backbone.base_forward(x)
        # FGSR feature-detach diagnostic: only the reliability branch sees
        # detached features; the segmentation decoder always uses `feats`.
        feats_fgsr = [f.detach() for f in feats] if self.detach_fgsr_features else feats

        if self.use_hybrid_decoder:
            if need_fp:
                feats_fp = [torch.cat((f, nn.Dropout2d(0.5)(f))) for f in feats]
                outs = self.hybrid_decoder(feats_fp, return_stats=False)
                outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
                out, out_fp = outs.chunk(2)
                if return_reliability:
                    if not hasattr(self, 'fgsr_head'):
                        raise RuntimeError('return_reliability=True but FGSR not enabled (use_fgsr=True?)')
                    R = self.fgsr_head(feats_fgsr)
                    R = F.interpolate(R, size=(h, w), mode="bilinear", align_corners=False)
                    return out, out_fp, R
                return out, out_fp
            out, stats = self.hybrid_decoder(feats, return_stats=True)
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
            if return_reliability:
                if not hasattr(self, 'fgsr_head'):
                    raise RuntimeError('return_reliability=True but FGSR not enabled (use_fgsr=True?)')
                R = self.fgsr_head(feats_fgsr)
                R = F.interpolate(R, size=(h, w), mode="bilinear", align_corners=False)
                return out, R
            # Store stats for logging (accessed from training loop)
            self._hybrid_stats = stats
            return out

        if self.use_maa_decoder:
            if need_fp:
                # Feature perturbation: double batch with Dropout2d on all 4 levels
                feats_fp = [torch.cat((f, nn.Dropout2d(0.5)(f))) for f in feats]
                outs = self.maa_decoder(feats_fp)
                outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
                out, out_fp = outs.chunk(2)
                if return_reliability:
                    if not hasattr(self, 'fgsr_head'):
                        raise RuntimeError('return_reliability=True but FGSR not enabled (use_fgsr=True?)')
                    R = self.fgsr_head(feats_fgsr)
                    R = F.interpolate(R, size=(h, w), mode="bilinear", align_corners=False)
                    return out, out_fp, R
                return out, out_fp

            out = self.maa_decoder(feats)
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
            if return_reliability:
                if not hasattr(self, 'fgsr_head'):
                    raise RuntimeError('return_reliability=True but FGSR not enabled (use_fgsr=True?)')
                R = self.fgsr_head(feats_fgsr)
                R = F.interpolate(R, size=(h, w), mode="bilinear", align_corners=False)
                return out, R
            return out

        # ---- Original DeepLabV3+ forward (use_maa_decoder=False) ----
        c1, c4 = feats[0], feats[-1]

        if need_fp:
            outs = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))),
                                torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)

            if return_reliability:
                if not hasattr(self, 'fgsr_head'):
                    raise RuntimeError('return_reliability=True but FGSR not enabled in config (use_fgsr=True?)')
                R = self.fgsr_head(feats_fgsr)
                R = F.interpolate(R, size=(h, w), mode="bilinear", align_corners=False)
                return out, out_fp, R
            return out, out_fp

        out = self._decode(c1, c4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        if return_reliability:
            if not hasattr(self, 'fgsr_head'):
                raise RuntimeError('return_reliability=True but FGSR not enabled in config (use_fgsr=True?)')
            R = self.fgsr_head(feats_fgsr)
            R = F.interpolate(R, size=(h, w), mode="bilinear", align_corners=False)
            return out, R
        return out

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)

        out = self.classifier(feature)

        return out


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)
