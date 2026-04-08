"""
TransRoadNet — Road Extraction via High-Level Semantic Feature and Context
PyTorch implementation based on:

  Yang et al., "TransRoadNet: A Novel Road Extraction Method for Remote
  Sensing Images via Combining High-Level Semantic Feature and Context"
  IEEE Geoscience and Remote Sensing Letters, Vol. 19, 2022.
  DOI: 10.1109/LGRS.2022.3171973

Architecture (Fig. 1 in paper)
───────────────────────────────
Input (3 ch, H×W)
  → ResNet34 encoder                 E1..E5 (strides ×2 each stage)
        E1: (64,  H/4,  W/4)    (conv7×7 stride2 + maxpool)
        E2: (64,  H/4,  W/4)    layer1 stride-1
        E3: (128, H/8,  W/8)    layer2
        E4: (256, H/16, W/16)   layer3
        E5: (512, H/32, W/32)   layer4

  → CIEM on E4 → FCI
        PA  : horizontal + vertical coordinate attention on E4
              → PA-weighted E4 feature map
        Swin: 2 layers, 3 heads → FCI  (B, 256, H/16, W/16)

  → FFM: p * FCI + (1-p) * E5 → Conv+BN+ReLU → FF   (B, 256, H/32, W/32)
         Note: FCI is at E4 resolution; E5 is spatially smaller.
         FF is produced at E5 resolution (both inputs projected to match).

  → FCISM at each skip (E2, E3, E4):
        Upsample FCI to match Ei resolution
        Concat(FCIi, Ei) → Conv+BN+ReLU → SSIi

  → Decoder (D5→D4→D3→D2→D1):
        Di = Upsample + Concat(SSIi) + Conv  → output

  → Head: 1×1 conv → (n_classes, H, W)

Key modules
───────────
  PositionAttention (PA)  — Coordinate-attention-style 1-D positional encoding
                            along H and W axes (Eq. 1–3, Fig. 2)

  MiniSwinBlock           — One Swin Transformer block with WMSA + SWMSA
                            (2 layers, 3 heads, window 4×4 as per §II-B)

  CIEM                    — PA → Swin → FCI

  FFM                     — Adaptive weighted sum p·FCI + (1-p)·E5,
                            then Conv+BN+ReLU  (Eq. 4, Fig. 4)

  FCISM                   — FCI upsampled + Ei → SSIi  (Eq. 5–6, Fig. 5)

  DecoderStage            — Upsample + Concat(SSIi) + Conv (Fig. 6)

Usage
─────
  from transroadnet import TransRoadNet

  model = TransRoadNet(n_classes=1, pretrained=True).to(device)
  logits = model(imgs)   # (B, 1, H, W)  raw logits — apply sigmoid/BCEWithLogitsLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


# ─────────────────────────────────────────────────────────────────────────────
#  Position Attention (PA) — horizontal + vertical coordinate attention
# ─────────────────────────────────────────────────────────────────────────────

class PositionAttention(nn.Module):
    """
    Position Attention module (§II-B, Eqs. 1–3, Fig. 2).

    Decomposes 2-D position attention into two independent 1-D directional
    codes (horizontal and vertical), inspired by coordinate attention.

    For each direction:
        1. Pool along the orthogonal axis → (B, C, H, 1) or (B, C, 1, W)
        2. AvgPool + MaxPool on the channel dim → two (B, 1, H) / (B, 1, W)
        3. Concat → Conv1×1 → Sigmoid → position weights Z_H or Z_W

    Final: X * Z_H(i) * Z_W(j)   (Eq. 3, element-wise broadcast)

    Args:
        channels : input feature channels
    """

    def __init__(self, channels: int):
        super().__init__()
        # Horizontal direction (reduce W, keep H)
        self.conv_h = nn.Sequential(
            nn.Conv2d(2, 1, 1, bias=False),   # input: cat(avg, max) → 2 ch
            nn.Sigmoid(),
        )
        # Vertical direction (reduce H, keep W)
        self.conv_w = nn.Sequential(
            nn.Conv2d(2, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, C, H, W)
        Returns:
            out : (B, C, H, W)  — position-weighted feature
        """
        # ── Horizontal weights Z_H  (pool across W axis) ──────────────────
        # F_c^H = mean over W → (B, C, H, 1)  [Eq. 1]
        fh = x.mean(dim=3, keepdim=True)            # (B, C, H, 1)
        # Avg/Max pool over channel dim → (B, 1, H, 1)
        fh_avg = fh.mean(dim=1, keepdim=True)
        fh_max, _ = fh.max(dim=1, keepdim=True)
        Z_H = self.conv_h(torch.cat([fh_avg, fh_max], dim=1))  # (B, 1, H, 1) [Eq. 2]

        # ── Vertical weights Z_W  (pool across H axis) ────────────────────
        fw = x.mean(dim=2, keepdim=True)            # (B, C, 1, W)
        fw_avg = fw.mean(dim=1, keepdim=True)
        fw_max, _ = fw.max(dim=1, keepdim=True)
        Z_W = self.conv_w(torch.cat([fw_avg, fw_max], dim=1))  # (B, 1, 1, W) [Eq. 2]

        # ── Apply positional weights (Eq. 3) ──────────────────────────────
        return x * Z_H * Z_W                        # broadcast over (H, W)


# ─────────────────────────────────────────────────────────────────────────────
#  Minimal Swin Transformer blocks
# ─────────────────────────────────────────────────────────────────────────────

class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention (W-MSA or SW-MSA).

    Args:
        dim       : channel dimension
        num_heads : number of attention heads
        window_sz : window size in pixels (paper uses 4×4 patches)
        shifted   : if True, apply cyclic shift (SW-MSA)
    """

    def __init__(self, dim: int, num_heads: int = 3, window_sz: int = 4,
                 shifted: bool = False):
        super().__init__()
        self.dim       = dim
        self.num_heads = num_heads
        self.window_sz = window_sz
        self.shifted   = shifted
        self.scale     = (dim // num_heads) ** -0.5
        self.shift     = window_sz // 2 if shifted else 0

        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        # Relative position bias table
        self.rel_pos_bias = nn.Parameter(
            torch.zeros((2 * window_sz - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)
        # Pre-compute relative position index
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_sz), torch.arange(window_sz), indexing='ij'
        ))                                                  # (2, Wz, Wz)
        coords_flat = coords.flatten(1)                    # (2, Wz²)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, Wz², Wz²)
        rel = rel.permute(1, 2, 0).contiguous()           # (Wz², Wz², 2)
        rel[:, :, 0] += window_sz - 1
        rel[:, :, 1] += window_sz - 1
        rel[:, :, 0] *= 2 * window_sz - 1
        self.register_buffer('rel_pos_idx', rel.sum(-1))   # (Wz², Wz²)

    @staticmethod
    def _window_partition(x: torch.Tensor, window_sz: int):
        """(B, H, W, C) → (B*nW, Wz, Wz, C)"""
        B, H, W, C = x.shape
        x = x.view(B, H // window_sz, window_sz,
                   W // window_sz, window_sz, C)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_sz, window_sz, C)

    @staticmethod
    def _window_reverse(windows: torch.Tensor, window_sz: int, H: int, W: int):
        """(B*nW, Wz, Wz, C) → (B, H, W, C)"""
        B = int(windows.shape[0] / (H * W / window_sz / window_sz))
        x = windows.view(B, H // window_sz, W // window_sz,
                         window_sz, window_sz, -1)
        return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, C, H, W)"""
        B, C, H, W = x.shape
        Wz = self.window_sz
        nh = self.num_heads

        # Pad to multiple of window_sz
        pad_h = (Wz - H % Wz) % Wz
        pad_w = (Wz - W % Wz) % Wz
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape

        x = x.permute(0, 2, 3, 1)   # (B, Hp, Wp, C)

        # Cyclic shift for SW-MSA
        if self.shifted:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))

        # Partition into windows
        wins = self._window_partition(x, Wz)            # (B*nW, Wz, Wz, C)
        wins_flat = wins.view(-1, Wz * Wz, C)           # (B*nW, Wz², C)

        # QKV
        qkv = self.qkv(wins_flat).chunk(3, dim=-1)      # each (B*nW, Wz², C)
        q, k, v = [t.view(-1, Wz * Wz, nh, C // nh).transpose(1, 2)
                   for t in qkv]                         # (B*nW, nh, Wz², dk)

        # Attention + relative position bias
        bias = self.rel_pos_bias[self.rel_pos_idx.view(-1)].view(
            Wz * Wz, Wz * Wz, nh).permute(2, 0, 1).unsqueeze(0)  # (1, nh, Wz², Wz²)
        attn = (q @ k.transpose(-2, -1)) * self.scale + bias
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(-1, Wz * Wz, C)
        out  = self.proj(out)

        # Reverse windows
        out = out.view(-1, Wz, Wz, C)
        out = self._window_reverse(out, Wz, Hp, Wp)     # (B, Hp, Wp, C)

        # Reverse cyclic shift
        if self.shifted:
            out = torch.roll(out, shifts=(self.shift, self.shift), dims=(1, 2))

        # Remove padding
        if pad_h or pad_w:
            out = out[:, :H, :W, :].contiguous()

        return out.permute(0, 3, 1, 2)                  # (B, C, H, W)


class SwinBlock(nn.Module):
    """
    One pair of W-MSA + SW-MSA Swin blocks (= one "layer" as listed in paper).

    Each block: LN → Attention → residual → LN → MLP → residual

    Args:
        dim       : channel dimension
        num_heads : number of attention heads (paper: 3)
        window_sz : window size in pixels (paper: 4×4)
        mlp_ratio : MLP hidden expansion
    """

    def __init__(self, dim: int, num_heads: int = 3,
                 window_sz: int = 4, mlp_ratio: int = 4):
        super().__init__()
        # W-MSA block
        self.ln1a   = nn.LayerNorm(dim)
        self.wmsa   = WindowAttention(dim, num_heads, window_sz, shifted=False)
        self.ln2a   = nn.LayerNorm(dim)
        self.mlp_a  = self._build_mlp(dim, mlp_ratio)

        # SW-MSA block
        self.ln1b   = nn.LayerNorm(dim)
        self.swmsa  = WindowAttention(dim, num_heads, window_sz, shifted=True)
        self.ln2b   = nn.LayerNorm(dim)
        self.mlp_b  = self._build_mlp(dim, mlp_ratio)

    @staticmethod
    def _build_mlp(dim, ratio):
        return nn.Sequential(
            nn.Linear(dim, dim * ratio),
            nn.GELU(),
            nn.Linear(dim * ratio, dim),
        )

    @staticmethod
    def _apply_ln(ln: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        return ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    @staticmethod
    def _apply_mlp(mlp: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        return mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # W-MSA
        x = x + self.wmsa(self._apply_ln(self.ln1a, x))
        x = x + self._apply_mlp(self.mlp_a, self._apply_ln(self.ln2a, x))
        # SW-MSA
        x = x + self.swmsa(self._apply_ln(self.ln1b, x))
        x = x + self._apply_mlp(self.mlp_b, self._apply_ln(self.ln2b, x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  CIEM  — Contextual Information Extraction Module
# ─────────────────────────────────────────────────────────────────────────────

class CIEM(nn.Module):
    """
    Contextual Information Extraction Module (§II-B, Fig. 2).

    Flow:
        E4  →  PA  →  X_pa  (position-weighted high-level features)
        X_pa → Swin (2 layers, 3 heads) → FCI

    Args:
        channels  : E4 channel count (256 for ResNet34)
        num_heads : heads in each Swin block (paper: 3)
        num_layers: number of SwinBlocks (paper: 2)
        window_sz : Swin window size
    """

    def __init__(self, channels: int = 256, num_heads: int = 3,
                 num_layers: int = 2, window_sz: int = 4):
        super().__init__()
        self.pa   = PositionAttention(channels)
        self.swin = nn.Sequential(*[
            SwinBlock(channels, num_heads, window_sz)
            for _ in range(num_layers)
        ])

    def forward(self, e4: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e4 : (B, 256, H/16, W/16)
        Returns:
            fci : (B, 256, H/16, W/16)
        """
        x = self.pa(e4)     # position-weighted features
        return self.swin(x) # FCI via swin transformer


# ─────────────────────────────────────────────────────────────────────────────
#  FFM  — Feature Fusion Module
# ─────────────────────────────────────────────────────────────────────────────

class FFM(nn.Module):
    """
    Feature Fusion Module (§II-C, Eq. 4, Fig. 4).

    FF = p * FCI + (1-p) * E5_proj   →  Conv+BN+ReLU  →  FF_out

    FCI is at E4 resolution (H/16); E5 is at (H/32).
    We upsample E5 to match FCI, then project both to a common channel size.

    Eq. 4:  FF = p × FCI + (1−p) × E5,   p ∈ [0,1]  (trainable)

    Args:
        fci_ch  : FCI channels (= E4 channels = 256)
        e5_ch   : E5 channels (= 512 for ResNet34)
        out_ch  : output channels (= 256)
    """

    def __init__(self, fci_ch: int = 256, e5_ch: int = 512, out_ch: int = 256):
        super().__init__()
        # Project E5 to fci_ch so we can add with FCI
        self.e5_proj = nn.Sequential(
            nn.Conv2d(e5_ch, fci_ch, 1, bias=False),
            nn.BatchNorm2d(fci_ch),
            nn.ReLU(inplace=True),
        )
        # Trainable scalar p (before sigmoid to stay in [0, 1])
        self.p_logit = nn.Parameter(torch.zeros(1))

        self.conv = nn.Sequential(
            nn.Conv2d(fci_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, fci: torch.Tensor, e5: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fci : (B, 256, H/16, W/16)
            e5  : (B, 512, H/32, W/32)
        Returns:
            ff  : (B, 256, H/16, W/16)
        """
        # Upsample E5 to match FCI spatial resolution
        e5_up = F.interpolate(e5, size=fci.shape[2:],
                              mode='bilinear', align_corners=False)
        e5_proj = self.e5_proj(e5_up)       # (B, 256, H/16, W/16)

        p = torch.sigmoid(self.p_logit)     # scalar ∈ (0, 1)
        ff = p * fci + (1.0 - p) * e5_proj # Eq. 4
        return self.conv(ff)                # (B, 256, H/16, W/16)


# ─────────────────────────────────────────────────────────────────────────────
#  FCISM  — FCI Supplement Module
# ─────────────────────────────────────────────────────────────────────────────

class FCISM(nn.Module):
    """
    Foreground Contextual Information Supplement Module (§II-D, Eqs. 5–6, Fig. 5).

    For each decoder stage i:
        FCIi = Conv(UP_ratio(FCI))    → match (H_i, W_i, C_i)   [Eq. 5]
        SSIi = ReLU(BN(Conv(Concat(FCIi, Ei))))                  [Eq. 6]

    This module is instantiated once per skip-connection level.

    Args:
        fci_ch  : FCI source channels (256)
        skip_ch : encoder skip Ei channel count
        out_ch  : SSI output channels (= skip_ch typically)
    """

    def __init__(self, fci_ch: int = 256, skip_ch: int = 128, out_ch: int = 128):
        super().__init__()
        # FCI projection: adjust channels to match skip
        self.fci_proj = nn.Sequential(
            nn.Conv2d(fci_ch, skip_ch, 1, bias=False),
            nn.BatchNorm2d(skip_ch),
            nn.ReLU(inplace=True),
        )
        # Fusion: Concat(FCIi, Ei) → Conv+BN+ReLU
        self.fuse = nn.Sequential(
            nn.Conv2d(skip_ch * 2, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, fci: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fci : (B, 256, H_fci, W_fci)   — FCI from CIEM (at E4 resolution)
            ei  : (B, skip_ch, H_i, W_i)   — encoder skip at stage i
        Returns:
            ssi : (B, out_ch, H_i, W_i)    — supplementary structure info
        """
        # Upsample + project FCI to match Ei resolution
        fci_up   = F.interpolate(fci, size=ei.shape[2:],
                                 mode='bilinear', align_corners=False)
        fci_proj = self.fci_proj(fci_up)              # (B, skip_ch, H_i, W_i)
        return self.fuse(torch.cat([fci_proj, ei], dim=1))   # SSIi  [Eq. 6]


# ─────────────────────────────────────────────────────────────────────────────
#  Decoder Stage
# ─────────────────────────────────────────────────────────────────────────────

class DecoderStage(nn.Module):
    """
    Single decoder stage (Fig. 6):
        Upsample ×2 → Concat with SSI → Conv3×3 + BN + ReLU × 2

    Args:
        in_ch  : channels from previous decoder output
        ssi_ch : SSI channels from FCISM at this resolution
        out_ch : output channels
    """

    def __init__(self, in_ch: int, ssi_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + ssi_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, ssi: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=ssi.shape[2:],
                          mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, ssi], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
#  Full TransRoadNet
# ─────────────────────────────────────────────────────────────────────────────

class TransRoadNet(nn.Module):
    """
    TransRoadNet (Yang et al., IEEE GRSL 2022).

    ResNet34 encoder → CIEM (PA + Swin) → FFM → FCISM × 4 → Decoder × 5 → Head

    Spatial flow (1024×1024 example, resnet strides):
      E1 : (B,  64, 256, 256)   conv7×7 stride2  (cumulative ×4 with maxpool)
      E2 : (B,  64, 256, 256)   layer1, stride=1  ← same as E1
      E3 : (B, 128, 128, 128)   layer2
      E4 : (B, 256,  64,  64)   layer3   ← input to CIEM
      E5 : (B, 512,  32,  32)   layer4   ← input to FFM

    Note: paper labels encoder outputs E1–E5 in its figure (Fig. 1), where
    E1 is after stem pool and before layer1, E2 after layer1, etc.  We use
    the same labelling.

    FCISM applied at E2, E3, E4 levels (three decoder stages need SSI).
    The first decoder stage (D5) merges FF with no SSI (just upsample+conv).

    Args:
        n_classes    : 1 for binary segmentation (raw logits)
        swin_heads   : attention heads in each Swin block (paper: 3)
        swin_layers  : number of Swin blocks in CIEM (paper: 2)
        swin_window  : Swin window size in pixels (paper: 4)
        pretrained   : load ImageNet weights for ResNet34 backbone
    """

    def __init__(
        self,
        n_classes:   int  = 1,
        swin_heads:  int  = 8,
        swin_layers: int  = 2,
        swin_window: int  = 4,
        pretrained:  bool = True,
    ):
        super().__init__()

        # ── ResNet34 Encoder ─────────────────────────────────────────────────
        weights  = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)

        # E1: 7×7 conv + bn + relu → maxpool   output: (64, H/4, W/4)
        self.stem  = nn.Sequential(backbone.conv1, backbone.bn1,
                                   backbone.relu, backbone.maxpool)
        self.enc2  = backbone.layer1    # (B,  64, H/4,  W/4)   stride 1
        self.enc3  = backbone.layer2    # (B, 128, H/8,  W/8)   stride 2
        self.enc4  = backbone.layer3    # (B, 256, H/16, W/16)  stride 2
        self.enc5  = backbone.layer4    # (B, 512, H/32, W/32)  stride 2

        # Channel sizes at each stage
        ch = dict(e2=64, e3=128, e4=256, e5=512)

        # ── CIEM (PA + Swin on E4) ───────────────────────────────────────────
        # self.ciem  = CIEM(channels=ch['e4'],
        #                   num_heads=swin_heads,
        #                   num_layers=swin_layers,
        #                   window_sz=swin_window)
        self.ciem = CIEM(channels=ch['e4'],
                 num_heads=8,            # 256÷8=32  ✓
                 num_layers=swin_layers,
                 window_sz=swin_window)

        # ── FFM (FCI + E5 → FF) ──────────────────────────────────────────────
        ff_out_ch  = 256
        self.ffm   = FFM(fci_ch=ch['e4'], e5_ch=ch['e5'], out_ch=ff_out_ch)

        # ── FCISM at E4, E3, E2 levels ────────────────────────────────────────
        self.fcism4 = FCISM(fci_ch=ch['e4'], skip_ch=ch['e4'], out_ch=ch['e4'])
        self.fcism3 = FCISM(fci_ch=ch['e4'], skip_ch=ch['e3'], out_ch=ch['e3'])
        self.fcism2 = FCISM(fci_ch=ch['e4'], skip_ch=ch['e2'], out_ch=ch['e2'])

        # ── Decoder ───────────────────────────────────────────────────────────
        # D5: FF (256, H/16) + SSI4 (256, H/16) → 256 at H/8
        #     (FF is already at H/16; SSI4 also at H/16 → upsample to H/8)
        self.dec5  = DecoderStage(ff_out_ch, ch['e4'], 256)   # FF + SSI4 → H/8
        # D4: 256 + SSI3 (128) → 128 at H/4
        self.dec4  = DecoderStage(256, ch['e3'], 128)
        # D3: 128 + SSI2 (64) → 64 at H/2
        self.dec3  = DecoderStage(128, ch['e2'], 64)
        # D2: 64, upsample to H/2 → H  (no more SSI, simple upsample+conv)
        self.dec2  = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ── Head ──────────────────────────────────────────────────────────────
        self.head = nn.Conv2d(32, n_classes, 1)

        self._init_new_weights()

    # ── Weight init (backbone is pretrained; init everything else) ────────────

    def _init_new_weights(self):
        backbone_ids = set(
            id(p) for m in [self.stem, self.enc2, self.enc3, self.enc4, self.enc5]
            for p in m.parameters()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if id(m.weight) not in backbone_ids:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                            nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, H, W)
        Returns:
            logits : (B, n_classes, H, W)  — no sigmoid
        """
        # ── Encoder ──────────────────────────────────────────────────────────
        e1 = self.stem(x)          # (B,  64, H/4,  W/4)
        e2 = self.enc2(e1)         # (B,  64, H/4,  W/4)
        e3 = self.enc3(e2)         # (B, 128, H/8,  W/8)
        e4 = self.enc4(e3)         # (B, 256, H/16, W/16)
        e5 = self.enc5(e4)         # (B, 512, H/32, W/32)

        # ── CIEM: PA + Swin → FCI ────────────────────────────────────────────
        fci = self.ciem(e4)        # (B, 256, H/16, W/16)

        # ── FFM: FCI + E5 → FF ───────────────────────────────────────────────
        ff  = self.ffm(fci, e5)    # (B, 256, H/16, W/16)

        # ── FCISM: FCI supplemented at each encoder skip ──────────────────────
        ssi4 = self.fcism4(fci, e4)   # (B, 256, H/16, W/16)
        ssi3 = self.fcism3(fci, e3)   # (B, 128, H/8,  W/8)
        ssi2 = self.fcism2(fci, e2)   # (B,  64, H/4,  W/4)

        # ── Decoder ──────────────────────────────────────────────────────────
        d = self.dec5(ff,  ssi4)   # up FF (H/16→H/8),  cat SSI4 → (B, 256, H/8)
        d = self.dec4(d,   ssi3)   # up (H/8→H/4),  cat SSI3    → (B, 128, H/4)
        d = self.dec3(d,   ssi2)   # up (H/4→H/2),  cat SSI2    → (B,  64, H/2)
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False)
        d = self.dec2(d)           #                               (B,  32, H)
        d = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False)  # ← add this

        return self.head(d)        # (B, n_classes, H, W)


# ─────────────────────────────────────────────────────────────────────────────
#  Sanity check  (python transroadnet.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}\n')

    model = TransRoadNet(n_classes=1, pretrained=False).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_params / 1e6:.2f} M')
    print(f'  (paper reports 68.21 M with pretrained ResNet34)\n')

    for name, module in [
        ('Stem+Enc',  [model.stem, model.enc2, model.enc3, model.enc4, model.enc5]),
        ('CIEM',      [model.ciem]),
        ('FFM',       [model.ffm]),
        ('FCISM×3',   [model.fcism4, model.fcism3, model.fcism2]),
        ('Decoder',   [model.dec5, model.dec4, model.dec3, model.dec2]),
        ('Head',      [model.head]),
    ]:
        n = sum(p.numel() for m in module for p in m.parameters() if p.requires_grad)
        print(f'  {name:<14} {n / 1e6:>6.2f} M')

    print()
    for H, W in [(256, 256), (512, 512)]:
        dummy  = torch.randn(2, 3, H, W, device=device)
        logits = model(dummy)
        assert logits.shape == (2, 1, H, W), f'Shape mismatch: {logits.shape}'
        print(f'Input {tuple(dummy.shape)}  →  Output {tuple(logits.shape)}  ✓')

    # Check FFM trainable parameter p
    p_val = torch.sigmoid(model.ffm.p_logit).item()
    print(f'\nFFM initial p = {p_val:.4f}  (should be ~0.5, i.e. sigmoid(0))')

    print('\nAll checks passed.')