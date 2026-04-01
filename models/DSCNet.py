# -*- coding: utf-8 -*-
"""
DSCNet — Dynamic Snake Convolution Network
Paper: "Dynamic Snake Convolution based on Topological Geometric Constraints
        for Tubular Structure Segmentation" — ICCV 2023
GitHub: https://github.com/YaoleiQi/DSCNet

Changes vs. original for this pipeline:
  • No `device` argument anywhere — fully device-agnostic via tensor.device
  • sigmoid REMOVED from forward() — CombinedLoss expects raw logits
  • GroupNorm guard (fallback to InstanceNorm) for odd channel counts
  • n_channels wired through (supports 4-channel LiDAR input)
  • `number` defaults to 8 (≈ 2.7 M params, fast); set to 16 for the
    paper's default capacity (≈ 10.7 M params)

Drop-in:
    from models.DSCNet import DSCNet

    model = DSCNet(n_channels=4, n_classes=1)
    logits = model(imgs)   # (B, 1, H, W) — raw logits, no sigmoid
"""

import torch
import torch.nn as nn
from models.DSConv import DSConv, _norm   # our cleaned DSConv


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _EncoderConv(nn.Module):
    """Standard 3×3 conv + norm + ReLU  (used in encoder)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            _norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _DecoderConv(nn.Module):
    """Standard 3×3 conv + norm + ReLU  (used in decoder)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            _norm(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _TriConv(nn.Module):
    """
    One DSCNet block = standard conv + DSConv-x + DSConv-y → concat → merge.
    This is the 'multi-view feature fusion' from §3.2 of the paper.

    channel flow:
        input (in_ch) →  standard:  out_ch
                       →  DSConv-x: out_ch
                       →  DSConv-y: out_ch
        concat → 3*out_ch → merge conv → out_ch
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        extend_scope: float,
        if_offset: bool,
        is_decoder: bool = False,
    ):
        super().__init__()
        ConvCls = _DecoderConv if is_decoder else _EncoderConv

        self.std  = ConvCls(in_ch, out_ch)
        self.dscx = DSConv(in_ch, out_ch, kernel_size, extend_scope, morph=0, if_offset=if_offset)
        self.dscy = DSConv(in_ch, out_ch, kernel_size, extend_scope, morph=1, if_offset=if_offset)
        self.merge = ConvCls(3 * out_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s  = self.std(x)
        dx = self.dscx(x)
        dy = self.dscy(x)
        return self.merge(torch.cat([s, dx, dy], dim=1))


# ---------------------------------------------------------------------------
# Full DSCNet (U-Net backbone with DSConv blocks)
# ---------------------------------------------------------------------------

class DSCNet(nn.Module):
    """
    DSCNet for binary segmentation of thin/tubular structures.

    Args:
        n_channels   : input image channels (4 for your LiDAR data)
        n_classes    : output channels (1 for binary segmentation)
        kernel_size  : snake kernel length — paper uses 9
        extend_scope : deformation range — paper default 1.0
        if_offset    : True = dynamic snake, False = straight (ablation)
        number       : base feature width
                         8  → ~2.7 M params  (fast, recommended to start)
                         16 → ~10.7 M params (paper default)

    Forward returns raw logits (B, n_classes, H, W).
    Apply sigmoid / use BCEWithLogitsLoss externally.
    """

    def __init__(
        self,
        n_channels: int = 4,
        n_classes:  int = 1,
        kernel_size:   int   = 9,
        extend_scope:  float = 1.0,
        if_offset:     bool  = True,
        number:        int   = 8,
    ):
        super().__init__()

        N = number   # shorthand

        kw = dict(kernel_size=kernel_size,
                  extend_scope=extend_scope,
                  if_offset=if_offset)

        # ── Encoder ────────────────────────────────────────────────────────
        # Block 0  (full resolution)
        self.enc0 = _TriConv(n_channels, N,  **kw)

        # Block 1  (½ resolution)
        self.enc1 = _TriConv(N,     2*N, **kw)

        # Block 2  (¼ resolution)
        self.enc2 = _TriConv(2*N,   4*N, **kw)

        # Block 3  (⅛ resolution — bottleneck)
        self.enc3 = _TriConv(4*N,   8*N, **kw)

        # ── Decoder ────────────────────────────────────────────────────────
        # Block 4  (¼ resolution, skip from enc2)
        self.dec2 = _TriConv(8*N + 4*N, 4*N, **kw, is_decoder=True)

        # Block 5  (½ resolution, skip from enc1)
        self.dec1 = _TriConv(4*N + 2*N, 2*N, **kw, is_decoder=True)

        # Block 6  (full resolution, skip from enc0)
        self.dec0 = _TriConv(2*N +   N,   N, **kw, is_decoder=True)

        # ── Head ───────────────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(N, n_classes, kernel_size=1)

        # ── Spatial ops ────────────────────────────────────────────────────
        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_channels, H, W)
        Returns:
            logits: (B, n_classes, H, W)  — NO sigmoid applied
        """
        # ── Encoder ────────────────────────────────────────────────────────
        s0 = self.enc0(x)               # (B,   N, H,    W)
        s1 = self.enc1(self.pool(s0))   # (B,  2N, H/2,  W/2)
        s2 = self.enc2(self.pool(s1))   # (B,  4N, H/4,  W/4)
        s3 = self.enc3(self.pool(s2))   # (B,  8N, H/8,  W/8)

        # ── Decoder ────────────────────────────────────────────────────────
        d2 = self.dec2(torch.cat([self.up(s3), s2], dim=1))  # (B, 4N, H/4,  W/4)
        d1 = self.dec1(torch.cat([self.up(d2), s1], dim=1))  # (B, 2N, H/2,  W/2)
        d0 = self.dec0(torch.cat([self.up(d1), s0], dim=1))  # (B,  N, H,    W)

        return self.out_conv(d0)   # (B, n_classes, H, W) — raw logits