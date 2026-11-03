import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """ (Conv + BN + ReLU) helper """
    def __init__(self, in_ch, out_ch, ks=1, padding=0):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, ks, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.seq(x)


class ASPPMAADecoder(nn.Module):
    """
    ASPP-MAA Hybrid Decoder.

    Keep the full DeepLabV3+ decoder (ASPP + c1 fuse) as the primary path,
    add a lightweight MAA gated multi-level branch as learnable residual.

    A = DeepLabV3+ path (c4→ASPP + c1→reduce) → 256ch, stride 4
    M = MAA path (c1/c2/c3/c4→gated fusion) → proj to 256ch, stride 4
    F = A + alpha * M
    logits = classifier(F)

    alpha: learnable scalar, init=0.1, enables gradual residual correction.
    """
    def __init__(self, num_classes, maa_dim=128):
        super().__init__()

        # === A path: DeepLabV3+ (c4→ASPP, c1→reduce, fuse) ===
        self.head = ASPPModule(2048, [6, 12, 18])
        self.reduce = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.fuse_a = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # === M path: MAA gated multi-level fusion ===
        self.proj_c1 = ConvBNReLU(256, maa_dim, 1)
        self.proj_c2 = ConvBNReLU(512, maa_dim, 1)
        self.proj_c3 = ConvBNReLU(1024, maa_dim, 1)
        self.proj_c4 = ConvBNReLU(2048, maa_dim, 1)

        self.gate_c1 = nn.Sequential(ConvBNReLU(256, maa_dim, 1), nn.Sigmoid())
        self.gate_c2 = nn.Sequential(ConvBNReLU(512, maa_dim, 1), nn.Sigmoid())
        self.gate_c3 = nn.Sequential(ConvBNReLU(1024, maa_dim, 1), nn.Sigmoid())
        self.gate_c4 = nn.Sequential(ConvBNReLU(2048, maa_dim, 1), nn.Sigmoid())

        self.fuse_m = nn.Sequential(
            nn.Conv2d(maa_dim, maa_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(maa_dim),
            nn.ReLU(inplace=True),
        )
        self.proj_m = nn.Conv2d(maa_dim, 256, 1)

        # === Learnable residual weight ===
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # === Classifier (shared, applied to F) ===
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, feats, return_stats=False):
        """ feats = [c1, c2, c3, c4] from backbone """
        c1, c2, c3, c4 = feats
        target_h, target_w = c2.shape[-2:]  # stride 8 (used for MAA alignment)
        h4, w4 = c1.shape[-2:]               # stride 4 (used for final upsample)

        # === A path: DeepLabV3+ ===
        c4_aspp = self.head(c4)                               # (B,256,15,20)
        c4_aspp = F.interpolate(c4_aspp, size=(h4, w4),       # (B,256,120,160)
                                mode='bilinear', align_corners=False)
        c1_r = self.reduce(c1)                                 # (B,48,120,160)
        A = self.fuse_a(torch.cat([c1_r, c4_aspp], dim=1))     # (B,256,120,160)

        # === M path: MAA gated fusion ===
        # Project + gate
        p1, g1 = self.proj_c1(c1), self.gate_c1(c1)           # (B,128,120,160)
        p2, g2 = self.proj_c2(c2), self.gate_c2(c2)           # (B,128,60,80)
        p3, g3 = self.proj_c3(c3), self.gate_c3(c3)           # (B,128,30,40)
        p4, g4 = self.proj_c4(c4), self.gate_c4(c4)           # (B,128,15,20)

        # Align to stride 8
        p1 = F.interpolate(p1, size=(target_h, target_w), mode='bilinear', align_corners=False)
        g1 = F.interpolate(g1, size=(target_h, target_w), mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, size=(target_h, target_w), mode='bilinear', align_corners=False)
        g3 = F.interpolate(g3, size=(target_h, target_w), mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=(target_h, target_w), mode='bilinear', align_corners=False)
        g4 = F.interpolate(g4, size=(target_h, target_w), mode='bilinear', align_corners=False)

        # Gated fusion
        fused_m = g1 * g2 * g3 * (p1 + p2) + p3 * g4 + p4    # (B,128,60,80)
        fused_m = self.fuse_m(fused_m)                         # (B,128,60,80)

        # Upsample to stride 4, project to 256ch
        fused_m = F.interpolate(fused_m, size=(h4, w4), mode='bilinear', align_corners=False)  # (B,128,120,160)
        M_feat = self.proj_m(fused_m)                          # (B,256,120,160)

        # === Residual fusion ===
        out_feat = A + self.alpha * M_feat                     # (B,256,120,160)

        # === Classifier ===
        logits = self.classifier(out_feat)                     # (B,num_cls,120,160)

        # === Stats for logging ===
        stats = {
            'alpha': self.alpha.item(),
            'A_mean': A.mean().item(),
            'A_std': A.std().item(),
            'M_mean': M_feat.mean().item(),
            'M_std': M_feat.std().item(),
            'alphaM_mean': (self.alpha * M_feat).mean().item(),
        }

        if return_stats:
            return logits, stats
        return logits


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super().__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rate1, dilation=rate1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rate2, dilation=rate2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=rate3, dilation=rate3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
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
