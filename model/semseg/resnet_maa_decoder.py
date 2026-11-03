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


class ResNetMAADecoder(nn.Module):
    """
    ResNet-adapted MAA Decoder.

    Gated multi-level feature fusion, inspired by UWSegFormer's
    InjectionMultiSumallmultiallsum but adapted for ResNet-50 channels.

    Input:  [c1(256), c2(512), c3(1024), c4(2048)] from ResNet-50 base_forward
    Output: logits (B, num_classes, H/8, W/8)

    Fusion (all aligned to stride 8):
        F = g1 * g2 * g3 * (p1 + p2) + p3 * g4 + p4
    """
    def __init__(self, num_classes, embed_dim=128):
        super().__init__()
        # ---- 1x1 projection per level ----
        self.proj_c1 = ConvBNReLU(256, embed_dim, 1)
        self.proj_c2 = ConvBNReLU(512, embed_dim, 1)
        self.proj_c3 = ConvBNReLU(1024, embed_dim, 1)
        self.proj_c4 = ConvBNReLU(2048, embed_dim, 1)

        # ---- 1x1 gate per level (ConvBNReLU + Sigmoid) ----
        self.gate_c1 = nn.Sequential(ConvBNReLU(256, embed_dim, 1), nn.Sigmoid())
        self.gate_c2 = nn.Sequential(ConvBNReLU(512, embed_dim, 1), nn.Sigmoid())
        self.gate_c3 = nn.Sequential(ConvBNReLU(1024, embed_dim, 1), nn.Sigmoid())
        self.gate_c4 = nn.Sequential(ConvBNReLU(2048, embed_dim, 1), nn.Sigmoid())

        # ---- Fusion head ----
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, feats):
        """ feats = [c1, c2, c3, c4] from backbone """
        c1, c2, c3, c4 = feats
        target_h, target_w = c2.shape[-2:]  # stride 8

        # ---- project + gate ----
        p1, g1 = self.proj_c1(c1), self.gate_c1(c1)
        p2, g2 = self.proj_c2(c2), self.gate_c2(c2)
        p3, g3 = self.proj_c3(c3), self.gate_c3(c3)
        p4, g4 = self.proj_c4(c4), self.gate_c4(c4)

        # ---- align to stride 8 ----
        p1 = F.interpolate(p1, size=(target_h, target_w), mode='bilinear', align_corners=False)
        g1 = F.interpolate(g1, size=(target_h, target_w), mode='bilinear', align_corners=False)
        # p2,g2 stay at stride 8
        p3 = F.interpolate(p3, size=(target_h, target_w), mode='bilinear', align_corners=False)
        g3 = F.interpolate(g3, size=(target_h, target_w), mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=(target_h, target_w), mode='bilinear', align_corners=False)
        g4 = F.interpolate(g4, size=(target_h, target_w), mode='bilinear', align_corners=False)

        # ---- gated multiply-add fusion ----
        # f = g1*g2*g3*(p1+p2) + p3*g4 + p4
        f = g1 * g2 * g3 * (p1 + p2) + p3 * g4 + p4

        # ---- fuse + classifier ----
        f = self.fuse(f)
        logits = self.classifier(f)

        return logits  # (B, num_classes, H/8, W/8)
