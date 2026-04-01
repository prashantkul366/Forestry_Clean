# models/FR_UNet.py
"""
FR-UNet — Full-Resolution UNet
Paper: "Full-Resolution Network and Dual-Threshold Iteration for Retinal
        Vessel and Coronary Angiograph Segmentation" — ISBI 2022
GitHub: https://github.com/lseventeen/FR-UNet

Key design idea (why it works for thin structures):
  Standard U-Nets lose thin-pixel information at every MaxPool step.
  FR-UNet keeps FULL resolution in block1_x — the first-scale path never
  downsamples, so no thin road pixel is ever discarded.  All other scales
  still downsample normally; their upsampled features are fused back via
  dense UNet++ style skip connections.

  Additionally, 5 deep supervision heads (one per column) are averaged at
  inference, which acts like an implicit ensemble and improves continuity.

Changes vs. original for this pipeline:
  • `from .utils import InitWeights_He` → `from models.fr_unet_utils import ...`
  • `num_channels` default changed from 1 → 4 (your LiDAR data)
  • `num_classes` default stays 1 (binary segmentation)
  • forward() returns raw logits — NO sigmoid — compatible with CombinedLoss
  • Added a brief forward-shape docstring for debugging

Usage:
    from models.FR_UNet import FR_UNet
    model = FR_UNet(num_channels=4, num_classes=1)
    logits = model(imgs)   # (B, 1, H, W) raw logits
"""

import torch
import torch.nn as nn
from models.fr_unet_utils import InitWeights_He


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class conv(nn.Module):
    """Double 3×3 conv + BN + Dropout + LeakyReLU block."""

    def __init__(self, in_c: int, out_c: int, dp: float = 0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.Dropout2d(dp),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class feature_fuse(nn.Module):
    """
    Multi-scale feature fusion: 1×1 + 3×3 + dilated 3×3 (rate=2) → sum + BN.
    Used as the channel-projection step when in_c != out_c.
    """

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv11    = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=False)
        self.conv33    = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.conv33_di = nn.Conv2d(in_c, out_c, kernel_size=3, padding=2,
                                   dilation=2, bias=False)
        self.norm = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.norm(self.conv11(x) + self.conv33(x) + self.conv33_di(x))


class up(nn.Module):
    """ConvTranspose2d ×2 upsample + BN + LeakyReLU."""

    def __init__(self, in_c: int, out_c: int, dp: float = 0):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=False),
        )

    def forward(self, x):
        return self.up(x)


class down(nn.Module):
    """Strided Conv2d ×2 downsample + BN + LeakyReLU."""

    def __init__(self, in_c: int, out_c: int, dp: float = 0):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.down(x)


class block(nn.Module):
    """
    Core FR-UNet block.

    Depending on (is_up, is_down) it returns:
        (False, False) → x
        (True,  False) → x, x_up
        (False, True)  → x, x_down
        (True,  True)  → x, x_up, x_down
    """

    def __init__(self, in_c: int, out_c: int, dp: float = 0,
                 is_up: bool = False, is_down: bool = False, fuse: bool = False):
        super().__init__()
        self.in_c  = in_c
        self.out_c = out_c
        self.is_up   = is_up
        self.is_down = is_down

        # channel projection (with multi-scale fusion if fuse=True)
        if in_c != out_c:
            self.fuse = feature_fuse(in_c, out_c) if fuse else \
                        nn.Conv2d(in_c, out_c, kernel_size=1)
        else:
            self.fuse = nn.Identity()

        self.conv = conv(out_c, out_c, dp=dp)

        if is_up:
            self.up_layer = up(out_c, out_c // 2)
        if is_down:
            self.down_layer = down(out_c, out_c * 2)

    def forward(self, x):
        if self.in_c != self.out_c:
            x = self.fuse(x)
        x = self.conv(x)

        if not self.is_up and not self.is_down:
            return x
        elif self.is_up and not self.is_down:
            return x, self.up_layer(x)
        elif not self.is_up and self.is_down:
            return x, self.down_layer(x)
        else:
            return x, self.up_layer(x), self.down_layer(x)


# ---------------------------------------------------------------------------
# Full FR-UNet
# ---------------------------------------------------------------------------

class FR_UNet(nn.Module):
    """
    Full-Resolution UNet for thin/tubular structure segmentation.

    Args:
        num_classes   : output channels  (1 for binary segmentation)
        num_channels  : input channels   (4 for your 4-channel LiDAR data)
        feature_scale : divide base filters [64,128,256,512,1024] by this
                        2 → [32,64,128,256,512] ≈ 9.5 M params  (default)
                        1 → full paper size ≈ 38 M params
        dropout       : spatial dropout probability
        fuse          : use multi-scale feature_fuse (True = paper default)
        out_ave       : average 5 deep supervision outputs (True = paper default)

    Returns raw logits (B, num_classes, H, W) — no sigmoid applied.
    """

    def __init__(
        self,
        num_classes:   int   = 1,
        num_channels:  int   = 4,
        feature_scale: int   = 2,
        dropout:       float = 0.2,
        fuse:          bool  = True,
        out_ave:       bool  = True,
    ):
        super().__init__()
        self.out_ave = out_ave

        f = [int(x / feature_scale) for x in [64, 128, 256, 512, 1024]]
        # f[0]=32, f[1]=64, f[2]=128, f[3]=256, f[4]=512  (with feature_scale=2)

        kw = dict(dp=dropout, fuse=fuse)

        # ── Column -3 to -1 (encoder, scale-1 path: FULL RESOLUTION) ─────────
        self.block1_3  = block(num_channels, f[0], is_down=True,  **kw)
        self.block1_2  = block(f[0],         f[0], is_down=True,  **kw)
        self.block1_1  = block(f[0]*2,       f[0], is_down=True,  **kw)

        # ── Column 0–3 (encoder-decoder transition) ───────────────────────────
        self.block10   = block(f[0]*2,       f[0], is_down=True,  **kw)
        self.block11   = block(f[0]*2,       f[0], is_down=True,  **kw)
        self.block12   = block(f[0]*2,       f[0],                **kw)
        self.block13   = block(f[0]*2,       f[0],                **kw)

        # ── Scale-2 path ──────────────────────────────────────────────────────
        self.block2_2  = block(f[1],         f[1], is_up=True, is_down=True, **kw)
        self.block2_1  = block(f[1]*2,       f[1], is_up=True, is_down=True, **kw)
        self.block20   = block(f[1]*3,       f[1], is_up=True, is_down=True, **kw)
        self.block21   = block(f[1]*3,       f[1], is_up=True,               **kw)
        self.block22   = block(f[1]*3,       f[1], is_up=True,               **kw)

        # ── Scale-3 path ──────────────────────────────────────────────────────
        self.block3_1  = block(f[2],         f[2], is_up=True, is_down=True, **kw)
        self.block30   = block(f[2]*2,       f[2], is_up=True,               **kw)
        self.block31   = block(f[2]*3,       f[2], is_up=True,               **kw)

        # ── Scale-4 path (deepest) ────────────────────────────────────────────
        self.block40   = block(f[3],         f[3], is_up=True,               **kw)

        # ── 5 deep supervision heads ──────────────────────────────────────────
        self.final1 = nn.Conv2d(f[0], num_classes, kernel_size=1, bias=True)
        self.final2 = nn.Conv2d(f[0], num_classes, kernel_size=1, bias=True)
        self.final3 = nn.Conv2d(f[0], num_classes, kernel_size=1, bias=True)
        self.final4 = nn.Conv2d(f[0], num_classes, kernel_size=1, bias=True)
        self.final5 = nn.Conv2d(f[0], num_classes, kernel_size=1, bias=True)

        # optional learned fusion of the 5 heads (not used when out_ave=True)
        self.fuse_head = nn.Conv2d(5, num_classes, kernel_size=1, bias=True)

        self.apply(InitWeights_He())

    def forward(self, x):
        """
        Args:
            x : (B, num_channels, H, W)
        Returns:
            logits : (B, num_classes, H, W)  — raw, no sigmoid
        
        Internal shapes  (with feature_scale=2, H=W=256):
            x1_3:  (B, 32,  256, 256)   x_down1_3: (B, 64,  128, 128)
            x2_2:  (B, 64,  128, 128)   x_up2_2:   (B, 32,  256, 256)
            x3_1:  (B, 128,  64,  64)   x_up3_1:   (B, 64,  128, 128)
            block40 produces x_up40:    (B, 128,  64,  64)
        """
        # ── Encoder / scale-1 full-res path ───────────────────────────────────
        x1_3,  x_down1_3  = self.block1_3(x)
        x1_2,  x_down1_2  = self.block1_2(x1_3)
        x2_2,  x_up2_2,  x_down2_2 = self.block2_2(x_down1_3)

        x1_1,  x_down1_1  = self.block1_1(torch.cat([x1_2,  x_up2_2],  dim=1))
        x2_1,  x_up2_1,  x_down2_1 = self.block2_1(torch.cat([x_down1_2, x2_2], dim=1))
        x3_1,  x_up3_1,  x_down3_1 = self.block3_1(x_down2_2)

        # ── Dense cross-scale fusion ───────────────────────────────────────────
        x10,   x_down10   = self.block10(torch.cat([x1_1,  x_up2_1],         dim=1))
        x20,   x_up20,   x_down20  = self.block20(torch.cat([x_down1_1, x2_1, x_up3_1], dim=1))
        x30,   x_up30    = self.block30(torch.cat([x_down2_1, x3_1],          dim=1))
        _,     x_up40    = self.block40(x_down3_1)

        x11,   x_down11   = self.block11(torch.cat([x10,  x_up20],            dim=1))
        x21,   x_up21    = self.block21(torch.cat([x_down10, x20, x_up30],    dim=1))
        _,     x_up31    = self.block31(torch.cat([x_down20, x30, x_up40],    dim=1))

        x12 = self.block12(torch.cat([x11, x_up21],                           dim=1))
        _,   x_up22    = self.block22(torch.cat([x_down11, x21, x_up31],      dim=1))
        x13 = self.block13(torch.cat([x12, x_up22],                           dim=1))

        # ── Output ────────────────────────────────────────────────────────────
        if self.out_ave:
            # average 5 deep supervision predictions (paper default)
            output = (self.final1(x1_1) + self.final2(x10) +
                      self.final3(x11)  + self.final4(x12) +
                      self.final5(x13)) / 5
        else:
            output = self.final5(x13)

        return output   # raw logits — CombinedLoss handles sigmoid internally