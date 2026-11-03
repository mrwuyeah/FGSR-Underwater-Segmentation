import torch
import torch.nn as nn
import torch.nn.functional as F


class FGSR(nn.Module):
    """FGSR for U2PL: multi-scale features → reliability map R ∈ [0.5, 1.0]"""
    def __init__(self, embed_dim=128, r_min=0.5):
        super().__init__()
        self.r_min = r_min
        # 1×1 conv projections - matches ResNet FPN output channels
        self.proj_c1 = nn.Conv2d(256, embed_dim, 1)
        self.proj_c2 = nn.Conv2d(512, embed_dim, 1)
        self.proj_c3 = nn.Conv2d(1024, embed_dim, 1)
        self.proj_c4 = nn.Conv2d(2048, embed_dim, 1)

        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Conv2d(embed_dim, 1, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feats):
        c1, c2, c3, c4 = feats  # [x1, x2, x3, x4] from encoder with FPN
        target_size = c1.shape[-2:]
        f1 = self.proj_c1(c1)
        f2 = F.interpolate(self.proj_c2(c2), size=target_size, mode='bilinear', align_corners=False)
        f3 = F.interpolate(self.proj_c3(c3), size=target_size, mode='bilinear', align_corners=False)
        f4 = F.interpolate(self.proj_c4(c4), size=target_size, mode='bilinear', align_corners=False)
        f = torch.cat([f1, f2, f3, f4], dim=1)
        f = self.fuse(f)
        R = self.conv_out(f)
        R = self.r_min + (1.0 - self.r_min) * torch.sigmoid(R)
        return R
