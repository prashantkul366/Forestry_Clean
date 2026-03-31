# =============================================================================
# EGE-UNet: Efficient Group Enhanced UNet (2023)
# Paper : https://arxiv.org/abs/2307.08473
# Origin: https://github.com/rzhangbq/EGE-UNet
#
# Architecture
#   Encoder : 6 stages of GMSC (Group Multi-Scale Convolution)
#             each stage halves spatial resolution
#   Bridge  : Group Aggregation Bridge (GAB) — optional skip gating
#   Decoder : symmetric 5-stage GMSC decoder with upsampling
#   Head    : 1×1 conv → num_classes
#
# Highlights
#   • ~50 K parameters (default c_list) — very lightweight
#   • Pure CNN, no transformers → no timm dependency
#   • Deep supervision available (set gt_ds=True + modify train loop)
#
# Install: only PyTorch required
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Primitives
# ─────────────────────────────────────────────────────────────────────────────

class GMSC(nn.Module):
    """
    Group Multi-Scale Convolution block.
    Applies depth-wise convs at 3 scales (3×3, 5×5, 7×7),
    concatenates, then point-wise fuses.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw3 = nn.Conv2d(in_ch, in_ch, 3, padding=1,  groups=in_ch, bias=False)
        self.dw5 = nn.Conv2d(in_ch, in_ch, 5, padding=2,  groups=in_ch, bias=False)
        self.dw7 = nn.Conv2d(in_ch, in_ch, 7, padding=3,  groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch * 3, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(
            torch.cat([self.dw3(x), self.dw5(x), self.dw7(x)], dim=1)
        )))


class GAB(nn.Module):
    """
    Group Aggregation Bridge.
    Upsamples the deeper feature, concatenates with skip,
    then applies a gated conv to produce the bridged output.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, deeper: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        deeper = F.interpolate(deeper, size=skip.shape[2:],
                               mode="bilinear", align_corners=False)
        x = self.conv(torch.cat([deeper, skip], dim=1))
        return x * self.gate(x)


# ─────────────────────────────────────────────────────────────────────────────
# EGE-UNet
# ─────────────────────────────────────────────────────────────────────────────

class EGEUNet(nn.Module):
    """
    EGE-UNet — Element-wise Gradient Enhancement UNet (2023)

    Args
    ----
    num_classes    : output channels  (1 for binary segmentation)
    input_channels : input channels
    c_list         : channel widths for the 6 encoder stages
                     default [8, 16, 24, 32, 48, 64] → ~50 K params
    bridge         : use Group Aggregation Bridges in skip connections
    gt_ds          : enable deep-supervision output heads
                     NOTE: when True + model.train(), forward() returns
                     (main_logit, aux5, aux4, aux3, aux2).
                     Your loss function must handle the tuple.
                     Set False (default) to keep the standard single-output API.
    """
    def __init__(
        self,
        num_classes    : int  = 1,
        input_channels : int  = 3,
        c_list         : list = None,
        bridge         : bool = True,
        gt_ds          : bool = False,
    ):
        super().__init__()
        if c_list is None:
            c_list = [8, 16, 24, 32, 48, 64]
        assert len(c_list) == 6, "c_list must have exactly 6 elements"

        self.bridge = bridge
        self.gt_ds  = gt_ds
        c           = c_list          # short alias

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = GMSC(input_channels, c[0])
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), GMSC(c[0], c[1]))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), GMSC(c[1], c[2]))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), GMSC(c[2], c[3]))
        self.enc5 = nn.Sequential(nn.MaxPool2d(2), GMSC(c[3], c[4]))
        self.enc6 = nn.Sequential(nn.MaxPool2d(2), GMSC(c[4], c[5]))  # bottleneck

        # ── Bridge ───────────────────────────────────────────────────────────
        if bridge:
            self.gab5 = GAB(c[5] + c[4], c[4])
            self.gab4 = GAB(c[4] + c[3], c[3])
            self.gab3 = GAB(c[3] + c[2], c[2])
            self.gab2 = GAB(c[2] + c[1], c[1])
            self.gab1 = GAB(c[1] + c[0], c[0])

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec5 = GMSC(c[5] + c[4], c[4])
        self.dec4 = GMSC(c[4] + c[3], c[3])
        self.dec3 = GMSC(c[3] + c[2], c[2])
        self.dec2 = GMSC(c[2] + c[1], c[1])
        self.dec1 = GMSC(c[1] + c[0], c[0])

        self.head = nn.Conv2d(c[0], num_classes, 1)

        # ── Deep supervision heads ────────────────────────────────────────────
        if gt_ds:
            self.ds5 = nn.Conv2d(c[4], num_classes, 1)
            self.ds4 = nn.Conv2d(c[3], num_classes, 1)
            self.ds3 = nn.Conv2d(c[2], num_classes, 1)
            self.ds2 = nn.Conv2d(c[1], num_classes, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _up_cat(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x: torch.Tensor):
        H, W = x.shape[2:]

        # Encode
        e1 = self.enc1(x)               # H,   W,   c[0]
        e2 = self.enc2(e1)              # H/2, W/2, c[1]
        e3 = self.enc3(e2)              # H/4
        e4 = self.enc4(e3)              # H/8
        e5 = self.enc5(e4)              # H/16
        e6 = self.enc6(e5)              # H/32  (bottleneck)

        # Bridge (optional skip gating)
        if self.bridge:
            s5 = self.gab5(e6, e5)
            s4 = self.gab4(s5, e4)
            s3 = self.gab3(s4, e3)
            s2 = self.gab2(s3, e2)
            s1 = self.gab1(s2, e1)
        else:
            s5, s4, s3, s2, s1 = e5, e4, e3, e2, e1

        # Decode
        d5 = self.dec5(self._up_cat(e6, s5))
        d4 = self.dec4(self._up_cat(d5, s4))
        d3 = self.dec3(self._up_cat(d4, s3))
        d2 = self.dec2(self._up_cat(d3, s2))
        d1 = self.dec1(self._up_cat(d2, s1))

        main = F.interpolate(self.head(d1), size=(H, W),
                             mode="bilinear", align_corners=False)

        # Deep supervision — only active during training
        if self.gt_ds and self.training:
            aux5 = F.interpolate(self.ds5(d5), (H, W), mode="bilinear", align_corners=False)
            aux4 = F.interpolate(self.ds4(d4), (H, W), mode="bilinear", align_corners=False)
            aux3 = F.interpolate(self.ds3(d3), (H, W), mode="bilinear", align_corners=False)
            aux2 = F.interpolate(self.ds2(d2), (H, W), mode="bilinear", align_corners=False)
            # Weighted sum → still a single tensor (drop-in with existing training loop)
            return main + 0.4 * aux5 + 0.3 * aux4 + 0.2 * aux3 + 0.1 * aux2

        return main