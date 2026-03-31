# =============================================================================
# DAEFormer: Dual Attention Enhanced Transformer (MICCAI 2023)
# Paper : https://arxiv.org/abs/2212.13504
# Origin: https://github.com/mindflow-institue/DAEFormer
#
# Architecture
#   Encoder : Mix-Transformer (same as SegFormer) — 4 hierarchical stages
#   Decoder : Dual Attention (Channel Attention + Spatial Attention) per scale,
#             then upsample-merge → prediction
#
# Install: pip install timm
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# ─────────────────────────────────────────────────────────────────────────────
# Mix-Transformer Encoder building blocks
# ─────────────────────────────────────────────────────────────────────────────

class DWConv(nn.Module):
    """Depth-wise conv used inside MLP to add local positional info."""
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MixMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        hidden = hidden_features or in_features
        self.fc1   = nn.Linear(in_features, hidden)
        self.dwconv = DWConv(hidden)
        self.act   = nn.GELU()
        self.fc2   = nn.Linear(hidden, in_features)
        self.drop  = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.drop(self.act(self.dwconv(self.fc1(x), H, W)))
        return self.drop(self.fc2(x))


class EfficientSelfAttention(nn.Module):
    """Efficient self-attention with sequence-reduction ratio `sr_ratio`."""
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.scale     = (dim // num_heads) ** -0.5
        self.q         = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv        = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio  = sr_ratio
        if sr_ratio > 1:
            self.sr   = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        h = self.num_heads
        q = self.q(x).reshape(B, N, h, C // h).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.norm(self.sr(x_).reshape(B, C, -1).permute(0, 2, 1))
            kv = self.kv(x_)
        else:
            kv = self.kv(x)

        kv = kv.reshape(B, -1, 2, h, C // h).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class MixTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., sr_ratio=1):
        super().__init__()
        self.norm1      = nn.LayerNorm(dim)
        self.attn       = EfficientSelfAttention(dim, num_heads, qkv_bias,
                                                  attn_drop, drop, sr_ratio)
        self.drop_path  = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2      = nn.LayerNorm(dim)
        self.mlp        = MixMlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding (SegFormer-style)."""
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=32):
        super().__init__()
        p = patch_size // 2
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride=stride, padding=p)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, H, W


# ─────────────────────────────────────────────────────────────────────────────
# Dual Attention (Channel + Spatial) — applied per-scale in the decoder
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        mid = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc       = nn.Sequential(
            nn.Conv2d(in_planes, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_planes, 1, bias=False),
        )
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class DualAttentionBlock(nn.Module):
    """Channel attention → Spatial attention (CBAM-style)."""
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# DAEFormer Decoder Head
# ─────────────────────────────────────────────────────────────────────────────

class DAEFormerHead(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim=256):
        super().__init__()
        self.proj  = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for c in in_channels
        ])
        self.da    = nn.ModuleList([DualAttentionBlock(embed_dim) for _ in in_channels])
        self.fuse  = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.head  = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features, out_hw):
        # All scales upsample to 4× the coarsest scale's spatial size
        target_hw = (features[0].shape[2] * 4, features[0].shape[3] * 4)
        outs = []
        for feat, proj, da in zip(features, self.proj, self.da):
            x = da(proj(feat))
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
            outs.append(x)
        x = self.fuse(torch.cat(outs, dim=1))
        x = self.head(x)
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# Full DAEFormer
# ─────────────────────────────────────────────────────────────────────────────

class DAEFormer(nn.Module):
    """
    DAEFormer — Dual Attention Enhanced Transformer (MICCAI 2023)

    Args
    ----
    num_classes    : output channels (1 for binary segmentation)
    input_channels : input image channels
    img_size       : spatial size (square) — used only for drop-path scheduling
    embed_dims     : feature dims per stage   [32, 64, 160, 256] → ~25 M params
    depths         : transformer blocks per stage
    num_heads      : attention heads per stage
    sr_ratios      : sequence-reduction ratios per stage (for efficiency)
    """
    def __init__(
        self,
        num_classes    = 1,
        input_channels = 3,
        img_size       = 224,
        embed_dims     = [32, 64, 160, 256],
        depths         = [2,  2,   2,   2],
        num_heads      = [1,  2,   5,   8],
        mlp_ratios     = [4,  4,   4,   4],
        qkv_bias       = True,
        sr_ratios      = [8,  4,   2,   1],
        drop_rate      = 0.,
        attn_drop_rate = 0.,
        drop_path_rate = 0.1,
        decoder_dim    = 256,
    ):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # ── Stage 1 ──────────────────────────────────────────────────────────
        self.patch1 = OverlapPatchEmbed(7, stride=4, in_chans=input_channels, embed_dim=embed_dims[0])
        self.blk1   = nn.ModuleList([MixTransformerBlock(embed_dims[0], num_heads[0], mlp_ratios[0],
                                      qkv_bias, drop_rate, attn_drop_rate, dpr[cur+i], sr_ratios[0])
                                      for i in range(depths[0])])
        self.norm1  = nn.LayerNorm(embed_dims[0])
        cur        += depths[0]

        # ── Stage 2 ──────────────────────────────────────────────────────────
        self.patch2 = OverlapPatchEmbed(3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.blk2   = nn.ModuleList([MixTransformerBlock(embed_dims[1], num_heads[1], mlp_ratios[1],
                                      qkv_bias, drop_rate, attn_drop_rate, dpr[cur+i], sr_ratios[1])
                                      for i in range(depths[1])])
        self.norm2  = nn.LayerNorm(embed_dims[1])
        cur        += depths[1]

        # ── Stage 3 ──────────────────────────────────────────────────────────
        self.patch3 = OverlapPatchEmbed(3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.blk3   = nn.ModuleList([MixTransformerBlock(embed_dims[2], num_heads[2], mlp_ratios[2],
                                      qkv_bias, drop_rate, attn_drop_rate, dpr[cur+i], sr_ratios[2])
                                      for i in range(depths[2])])
        self.norm3  = nn.LayerNorm(embed_dims[2])
        cur        += depths[2]

        # ── Stage 4 ──────────────────────────────────────────────────────────
        self.patch4 = OverlapPatchEmbed(3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.blk4   = nn.ModuleList([MixTransformerBlock(embed_dims[3], num_heads[3], mlp_ratios[3],
                                      qkv_bias, drop_rate, attn_drop_rate, dpr[cur+i], sr_ratios[3])
                                      for i in range(depths[3])])
        self.norm4  = nn.LayerNorm(embed_dims[3])

        # ── Decoder ──────────────────────────────────────────────────────────
        self.head   = DAEFormerHead(embed_dims, num_classes, embed_dim=decoder_dim)

        self.apply(self._init_weights)

    # ── weight init ──────────────────────────────────────────────────────────
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None: nn.init.zeros_(m.bias)

    # ── one encoder stage ────────────────────────────────────────────────────
    def _run_stage(self, x, patch_embed, blocks, norm):
        x, H, W = patch_embed(x)
        for blk in blocks:
            x = blk(x, H, W)
        B, _, C = x.shape
        return norm(x).reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        H, W = x.shape[2:]
        f1 = self._run_stage(x,  self.patch1, self.blk1, self.norm1)   # H/4
        f2 = self._run_stage(f1, self.patch2, self.blk2, self.norm2)   # H/8
        f3 = self._run_stage(f2, self.patch3, self.blk3, self.norm3)   # H/16
        f4 = self._run_stage(f3, self.patch4, self.blk4, self.norm4)   # H/32
        return self.head([f1, f2, f3, f4], (H, W))