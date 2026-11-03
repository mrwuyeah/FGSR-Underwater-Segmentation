import torch
import torch.nn as nn
import torch.nn.functional as F


class FGSR(nn.Module):
    """
    Feature-Guided Spatial Reweighting.

    Takes multi-scale backbone features [c1, c2, c3, c4] and produces a
    pixel-level reliability map R ∈ [0.5, 1.0] for the strong-view prediction.

    Feature channels (ResNet-50):
        c1: 256ch  stride 4
        c2: 512ch  stride 8
        c3: 1024ch stride 16
        c4: 2048ch stride 32

    Design: SegFormer-style MLP projection → upsample → concat → fuse → 1ch sigmoid.
    """
    def __init__(self, embed_dim=128, r_min=0.5):
        super().__init__()
        self.r_min = r_min
        # 1×1 conv projection per level
        self.proj_c1 = nn.Conv2d(256, embed_dim, 1)
        self.proj_c2 = nn.Conv2d(512, embed_dim, 1)
        self.proj_c3 = nn.Conv2d(1024, embed_dim, 1)
        self.proj_c4 = nn.Conv2d(2048, embed_dim, 1)

        # Fusion: 3×3 conv + BN + ReLU
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # Output: 1ch reliability logits
        self.conv_out = nn.Conv2d(embed_dim, 1, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feats):
        """ feats: [c1, c2, c3, c4] from backbone.base_forward(x)
        Returns:
            R: (B, 1, H/4, W/4), R ∈ [r_min, 1.0]
        """
        c1, c2, c3, c4 = feats
        target_size = c1.shape[-2:]  # H/4, W/4

        # 1×1 conv projection + upsample to target resolution
        f1 = self.proj_c1(c1)
        f2 = F.interpolate(self.proj_c2(c2), size=target_size,
                           mode='bilinear', align_corners=False)
        f3 = F.interpolate(self.proj_c3(c3), size=target_size,
                           mode='bilinear', align_corners=False)
        f4 = F.interpolate(self.proj_c4(c4), size=target_size,
                           mode='bilinear', align_corners=False)

        # Concat + fuse
        f = torch.cat([f1, f2, f3, f4], dim=1)
        f = self.fuse(f)

        # Reliability map: R ∈ [r_min, 1.0]
        R = self.conv_out(f)
        R = self.r_min + (1.0 - self.r_min) * torch.sigmoid(R)

        return R  # (B, 1, H/4, W/4)
