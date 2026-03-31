# =============================================================================
# VM-UNet: Vision Mamba UNet (2024)
# Paper : https://arxiv.org/abs/2402.02491
# Origin: https://github.com/JCruan519/VM-UNet
#
# Architecture
#   Encoder : 4 patch-merging stages, each with N × VSSBlock
#   Decoder : 4 patch-expanding stages with skip connections
#   Core op : Visual State Space (VSS) block via Mamba SSM
#             applied bidirectionally (forward + reverse scan)
#
# Install:
#   pip install mamba-ssm causal-conv1d
#   (requires CUDA; for CPU-only testing use the lightweight fallback below)
#
# Notes
#   • If mamba-ssm is unavailable, VMUNet falls back to a plain
#     bidirectional GRU block so the rest of your code still runs.
#     The fallback is slower and less expressive — install mamba-ssm
#     for the real model.
#   • Default channel list matches the paper's VM-UNet-S variant.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

try:
    from mamba_ssm import Mamba
    _MAMBA_OK = True
except ImportError:
    _MAMBA_OK = False
    import warnings
    warnings.warn(
        "\n⚠️  mamba-ssm not found — VMUNet will use a GRU fallback.\n"
        "   Install the real Mamba with: pip install mamba-ssm causal-conv1d\n"
        "   (requires CUDA toolkit at install time)\n",
        stacklevel=2,
    )

try:
    from timm.models.layers import DropPath, trunc_normal_
except ImportError:
    from torch.nn import Identity as DropPath            # minimal fallback
    def trunc_normal_(t, std=.02): nn.init.normal_(t, std=std)


# ─────────────────────────────────────────────────────────────────────────────
# VSS Block (core of VMamba)
# ─────────────────────────────────────────────────────────────────────────────

class _GRUFallback(nn.Module):
    """Bidirectional GRU that mimics the Mamba interface."""
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model // 2, batch_first=True,
                          bidirectional=True)
    def forward(self, x):
        out, _ = self.gru(x)
        return out


class VSSBlock(nn.Module):
    """
    Visual State Space Block.
    • Norm → expand → depthwise conv → SiLU → Mamba (fwd+bwd) → gate → project
    • Residual connection around the whole block
    """
    def __init__(self, d_model: int, drop_path: float = 0.,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = int(d_model * expand)
        self.norm   = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)  # x + z

        # Mamba or GRU fallback
        _SSM = partial(Mamba, d_state=d_state, d_conv=d_conv, expand=1) \
               if _MAMBA_OK else _GRUFallback
        self.ssm_fwd = _SSM(d_model=d_inner)
        self.ssm_bwd = _SSM(d_model=d_inner)

        self.act      = nn.SiLU()
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        shortcut    = x
        xz          = self.in_proj(self.norm(x))            # [B, H, W, 2*inner]
        xz          = xz.view(B, H * W, -1)
        x_inner, z  = xz.chunk(2, dim=-1)                  # each [B, L, inner]

        # bidirectional scan
        fwd = self.ssm_fwd(x_inner)
        bwd = self.ssm_bwd(x_inner.flip(1)).flip(1)
        x_inner = self.act(fwd + bwd) * self.act(z)        # gating

        x_inner = self.out_proj(x_inner)                    # [B, L, C]
        x_inner = x_inner.view(B, H, W, C)
        return shortcut + self.drop_path(x_inner)


# ─────────────────────────────────────────────────────────────────────────────
# Patch Merging / Expanding
# ─────────────────────────────────────────────────────────────────────────────

class PatchMerge(nn.Module):
    """2× spatial downsampling via pixel-unshuffle + linear."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm  = nn.LayerNorm(in_ch * 4)
        self.proj  = nn.Linear(in_ch * 4, out_ch, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                        x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)  # [B, H/2, W/2, 4C]
        return self.proj(self.norm(x))


class PatchExpand(nn.Module):
    """2× spatial upsampling via linear + pixel-shuffle."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_ch)
        self.proj = nn.Linear(in_ch, out_ch * 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = self.proj(self.norm(x))               # [B, H, W, out*4]
        x = x.view(B, H, W, 2, 2, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B, H * 2, W * 2, -1)


class FinalExpand(nn.Module):
    """4× final upsampling for the output head."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_ch)
        self.proj = nn.Linear(in_ch, out_ch * 16, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = self.proj(self.norm(x))
        x = x.view(B, H, W, 4, 4, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B, H * 4, W * 4, -1)


# ─────────────────────────────────────────────────────────────────────────────
# VM-UNet
# ─────────────────────────────────────────────────────────────────────────────

class VMUNet(nn.Module):
    """
    VM-UNet — Vision Mamba UNet (2024)

    Args
    ----
    input_channels : input image channels
    num_classes    : output channels  (1 for binary seg)
    dims           : channel widths at each encoder stage
                     paper default (VM-UNet-S): [96, 192, 384, 768]
                     lighter variant             : [48,  96, 192, 384]
    depths         : VSS blocks per encoder stage
    depths_decoder : VSS blocks per decoder stage (reversed)
    drop_path_rate : stochastic depth rate
    """
    def __init__(
        self,
        input_channels : int   = 3,
        num_classes    : int   = 1,
        dims           : list  = None,
        depths         : list  = None,
        depths_decoder : list  = None,
        drop_path_rate : float = 0.1,
    ):
        super().__init__()

        if dims           is None: dims           = [48,  96, 192, 384]
        if depths         is None: depths         = [2,   2,  9,   2]
        if depths_decoder is None: depths_decoder = [2,   9,  2,   2]

        assert len(dims) == 4 and len(depths) == 4 and len(depths_decoder) == 4

        total_depth = sum(depths) + sum(depths_decoder)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        cur = 0

        # ── Stem: patch embed (4× downsample) ────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm([dims[0], 1, 1]),          # channel-wise norm placeholder
        )
        # Rewrite stem with proper norm
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=4, stride=4),
        )
        self.stem_norm = nn.LayerNorm(dims[0])

        # ── Encoder stages ────────────────────────────────────────────────────
        self.enc_layers = nn.ModuleList()
        self.downsample = nn.ModuleList()
        in_dim = dims[0]
        for i, (out_dim, d) in enumerate(zip(dims, depths)):
            stage = nn.ModuleList([
                VSSBlock(in_dim, dpr[cur + j]) for j in range(d)
            ])
            self.enc_layers.append(stage)
            cur += d
            if i < 3:  # no downsample after last stage
                self.downsample.append(PatchMerge(in_dim, out_dim))
                in_dim = out_dim
            else:
                self.downsample.append(nn.Identity())

        # ── Decoder stages ────────────────────────────────────────────────────
        self.dec_layers = nn.ModuleList()
        self.upsample   = nn.ModuleList()
        dec_dims = list(reversed(dims))  # [384, 192, 96, 48]
        for i, (d_dim, d) in enumerate(zip(dec_dims, depths_decoder)):
            if i < 3:
                up_out = dec_dims[i + 1]
                self.upsample.append(PatchExpand(d_dim, up_out))
                skip_dim = up_out
            else:
                self.upsample.append(nn.Identity())
                skip_dim = dec_dims[i]

            # after concat with skip → channel is skip_dim + skip from encoder
            merge_dim = skip_dim + dec_dims[i + 1] if i < 3 else d_dim
            stage = nn.ModuleList([
                VSSBlock(merge_dim if j == 0 else skip_dim, dpr[cur + j])
                for j in range(d)
            ])
            self.dec_layers.append(stage)
            cur += d

        # ── Final expand + head ───────────────────────────────────────────────
        self.final_up = FinalExpand(dims[0], dims[0])
        self.head     = nn.Conv2d(dims[0], num_classes, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Stem: [B, dims[0], H/4, W/4] → [B, H/4, W/4, dims[0]]
        x = self.stem(x).permute(0, 2, 3, 1)
        x = self.stem_norm(x)

        # Encoder: collect skip features
        skips = []
        for stage, down in zip(self.enc_layers, self.downsample):
            for blk in stage:
                x = blk(x)
            skips.append(x)
            if not isinstance(down, nn.Identity):
                x = down(x)

        # Decoder: upsample + skip-cat
        skips = list(reversed(skips))  # coarsest first for decoder
        for i, (stage, up) in enumerate(zip(self.dec_layers, self.upsample)):
            if not isinstance(up, nn.Identity):
                x = up(x)
                skip = skips[i + 1]
                x = torch.cat([x, skip], dim=-1)
            for j, blk in enumerate(stage):
                x = blk(x)

        # Final expand: [B, H/4, W/4, C] → [B, H, W, C]
        x = self.final_up(x)
        # [B, H, W, C] → [B, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.head(x)
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)