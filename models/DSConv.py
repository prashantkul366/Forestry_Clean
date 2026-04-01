# -*- coding: utf-8 -*-
"""
Dynamic Snake Convolution (DSConv)
Paper: "Dynamic Snake Convolution based on Topological Geometric Constraints
        for Tubular Structure Segmentation" — ICCV 2023
GitHub: https://github.com/YaoleiQi/DSCNet

Cleaned for drop-in use:
  • No explicit `device` argument — uses tensor.device throughout
  • GroupNorm guard: falls back to InstanceNorm when channels % 4 != 0
"""

import torch
import torch.nn as nn


def _norm(channels: int) -> nn.Module:
    """GroupNorm when divisible by 4, else InstanceNorm."""
    if channels % 4 == 0:
        return nn.GroupNorm(channels // 4, channels)
    return nn.InstanceNorm2d(channels, affine=True)


class DSConv(nn.Module):
    """
    Dynamic Snake Convolution.

    Args:
        in_ch       : input channels
        out_ch      : output channels
        kernel_size : length of the snake kernel (paper uses 9)
        extend_scope: deformation range multiplier (paper default = 1)
        morph       : 0 → snake along x-axis, 1 → snake along y-axis
        if_offset   : True = deformable snake, False = straight kernel
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        extend_scope: float = 1.0,
        morph: int = 0,
        if_offset: bool = True,
    ):
        super().__init__()

        # learns the per-pixel offsets that bend the snake
        self.offset_conv = nn.Conv2d(in_ch, 2 * kernel_size, 3, padding=1)
        self.bn = nn.BatchNorm2d(2 * kernel_size)
        self.kernel_size = kernel_size

        # two 1-D convolutions — one per axis
        self.dsc_conv_x = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=(kernel_size, 1),
            stride=(kernel_size, 1),
            padding=0,
        )
        self.dsc_conv_y = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, kernel_size),
            padding=0,
        )

        self.norm = _norm(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        offset = self.offset_conv(f)
        offset = self.bn(offset)
        offset = torch.tanh(offset)           # keep offsets in [-1, 1]

        dsc = _DSC(f.shape, self.kernel_size, self.extend_scope,
                   self.morph, f.device)
        deformed = dsc.deform_conv(f, offset, self.if_offset)

        if self.morph == 0:
            x = self.dsc_conv_x(deformed)
        else:
            x = self.dsc_conv_y(deformed)

        x = self.norm(x)
        x = self.relu(x)
        return x


# ---------------------------------------------------------------------------
# Internal geometry helper — builds the coordinate map and interpolates
# ---------------------------------------------------------------------------

class _DSC:
    """
    Coordinate-map builder and bilinear sampler for DSConv.
    All tensors are created on `device` passed from the feature map.
    """

    def __init__(self, input_shape, kernel_size, extend_scope, morph, device):
        self.num_points   = kernel_size
        self.width        = input_shape[2]   # H
        self.height       = input_shape[3]   # W
        self.morph        = morph
        self.device       = device
        self.extend_scope = extend_scope
        self.num_batch    = input_shape[0]
        self.num_channels = input_shape[1]

    # ------------------------------------------------------------------
    # Coordinate map  →  (y_coord_map, x_coord_map)
    # ------------------------------------------------------------------
    def _coordinate_map_3D(self, offset, if_offset):
        y_offset, x_offset = torch.split(offset, self.num_points, dim=1)

        # base grids (H × W)
        y_center = (torch.arange(0, self.width, device=self.device)
                        .repeat(self.height)
                        .reshape(self.height, self.width)
                        .permute(1, 0)
                        .reshape(-1, self.width, self.height)
                        .repeat(self.num_points, 1, 1)
                        .float()
                        .unsqueeze(0))

        x_center = (torch.arange(0, self.height, device=self.device)
                        .repeat(self.width)
                        .reshape(self.width, self.height)
                        .reshape(-1, self.width, self.height)
                        .repeat(self.num_points, 1, 1)
                        .float()
                        .unsqueeze(0))

        if self.morph == 0:
            # snake runs along x-axis: y stays 0, x spans [-K//2 .. K//2]
            y = torch.zeros(1, device=self.device)
            x = torch.linspace(
                -self.num_points // 2, self.num_points // 2,
                self.num_points, device=self.device)
            y, x = torch.meshgrid(y, x, indexing="ij")

            y_grid = y.reshape(-1, 1).repeat(1, self.width * self.height)
            y_grid = y_grid.reshape(self.num_points, self.width, self.height).unsqueeze(0)

            x_grid = x.reshape(-1, 1).repeat(1, self.width * self.height)
            x_grid = x_grid.reshape(self.num_points, self.width, self.height).unsqueeze(0)

            y_new = (y_center + y_grid).repeat(self.num_batch, 1, 1, 1)
            x_new = (x_center + x_grid).repeat(self.num_batch, 1, 1, 1)

            if if_offset:
                # cumulative offset: each step adds to the previous position
                y_off = y_offset.permute(1, 0, 2, 3)
                y_off_new = y_off.detach().clone()
                center = self.num_points // 2
                y_off_new[center] = 0
                for i in range(1, center):
                    y_off_new[center + i] = y_off_new[center + i - 1] + y_off[center + i]
                    y_off_new[center - i] = y_off_new[center - i + 1] + y_off[center - i]
                y_off_new = y_off_new.permute(1, 0, 2, 3)
                y_new = y_new + y_off_new * self.extend_scope

            # reshape to feed into bilinear sampler
            y_new = (y_new.reshape(self.num_batch, self.num_points, 1,
                                   self.width, self.height)
                          .permute(0, 3, 1, 4, 2)
                          .reshape(self.num_batch,
                                   self.num_points * self.width,
                                   self.height))
            x_new = (x_new.reshape(self.num_batch, self.num_points, 1,
                                   self.width, self.height)
                          .permute(0, 3, 1, 4, 2)
                          .reshape(self.num_batch,
                                   self.num_points * self.width,
                                   self.height))

        else:
            # snake runs along y-axis
            y = torch.linspace(
                -self.num_points // 2, self.num_points // 2,
                self.num_points, device=self.device)
            x = torch.zeros(1, device=self.device)
            y, x = torch.meshgrid(y, x, indexing="ij")

            y_grid = y.reshape(-1, 1).repeat(1, self.width * self.height)
            y_grid = y_grid.reshape(self.num_points, self.width, self.height).unsqueeze(0)

            x_grid = x.reshape(-1, 1).repeat(1, self.width * self.height)
            x_grid = x_grid.reshape(self.num_points, self.width, self.height).unsqueeze(0)

            y_new = (y_center + y_grid).repeat(self.num_batch, 1, 1, 1)
            x_new = (x_center + x_grid).repeat(self.num_batch, 1, 1, 1)

            if if_offset:
                x_off = x_offset.permute(1, 0, 2, 3)
                x_off_new = x_off.detach().clone()
                center = self.num_points // 2
                x_off_new[center] = 0
                for i in range(1, center):
                    x_off_new[center + i] = x_off_new[center + i - 1] + x_off[center + i]
                    x_off_new[center - i] = x_off_new[center - i + 1] + x_off[center - i]
                x_off_new = x_off_new.permute(1, 0, 2, 3)
                x_new = x_new + x_off_new * self.extend_scope

            y_new = (y_new.reshape(self.num_batch, 1, self.num_points,
                                   self.width, self.height)
                          .permute(0, 3, 1, 4, 2)
                          .reshape(self.num_batch,
                                   self.width,
                                   self.num_points * self.height))
            x_new = (x_new.reshape(self.num_batch, 1, self.num_points,
                                   self.width, self.height)
                          .permute(0, 3, 1, 4, 2)
                          .reshape(self.num_batch,
                                   self.width,
                                   self.num_points * self.height))

        return y_new, x_new

    # ------------------------------------------------------------------
    # Bilinear interpolation on the deformed grid
    # ------------------------------------------------------------------
    def _bilinear_interpolate_3D(self, feat, y, x):
        y = y.reshape(-1).float()
        x = x.reshape(-1).float()

        max_y = self.width  - 1
        max_x = self.height - 1

        y0 = torch.floor(y).long().clamp(0, max_y)
        y1 = (y0 + 1)     .clamp(0, max_y)
        x0 = torch.floor(x).long().clamp(0, max_x)
        x1 = (x0 + 1)     .clamp(0, max_x)

        # flat index into [B, C, H, W] → [B, H, W, C] → [B*H*W, C]
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, self.num_channels)
        dim = self.height * self.width

        base = (torch.arange(self.num_batch, device=self.device) * dim
                ).float().unsqueeze(1)
        repeat = torch.ones(
            self.num_points * self.width * self.height,
            device=self.device).unsqueeze(0).float()
        base = (base @ repeat).reshape(-1).long()

        def _gather(b_y, b_x):
            idx = (base + b_y * self.height + b_x).clamp(
                0, self.num_batch * dim - 1)
            return feat_flat[idx]

        va0 = _gather(y0, x0)
        va1 = _gather(y0, x1)
        vb0 = _gather(y1, x0)
        vb1 = _gather(y1, x1)

        y0f, y1f = y0.float(), y1.float()
        x0f, x1f = x0.float(), x1.float()

        w_a0 = ((y1f - y) * (x1f - x)).unsqueeze(-1)
        w_a1 = ((y1f - y) * (x  - x0f)).unsqueeze(-1)
        w_b0 = ((y  - y0f) * (x1f - x)).unsqueeze(-1)
        w_b1 = ((y  - y0f) * (x  - x0f)).unsqueeze(-1)

        out = va0 * w_a0 + va1 * w_a1 + vb0 * w_b0 + vb1 * w_b1

        if self.morph == 0:
            out = (out.reshape(self.num_batch,
                               self.num_points * self.width,
                               self.height,
                               self.num_channels)
                      .permute(0, 3, 1, 2))
        else:
            out = (out.reshape(self.num_batch,
                               self.width,
                               self.num_points * self.height,
                               self.num_channels)
                      .permute(0, 3, 1, 2))
        return out

    def deform_conv(self, feat, offset, if_offset):
        y, x = self._coordinate_map_3D(offset, if_offset)
        return self._bilinear_interpolate_3D(feat, y, x)