"""
C2S-RoadNet — Road Extraction with Depth-Wise Separable Convolution and Self-Attention
PyTorch implementation based on:

  Yin et al., "C2S-RoadNet: Road Extraction Model with Depth-Wise Separable
  Convolution and Self-Attention"
  Remote Sensing, 2023, 15, 4531.
  DOI: 10.3390/rs15184531

Architecture (Fig. 4 in paper)
────────────────────────────────
Input (3 ch, H×W)
  → Enc0  : Conv3×3 × 2 + MaxPool          →  (64,  H/2,  W/2)
  → Enc1  : DS2C block + MaxPool            → (128,  H/4,  W/4)
  → Enc2  : DS2C block + MaxPool            → (256,  H/8,  W/8)
  → Enc3  : DS2C block + MaxPool            → (512, H/16, W/16)
  → Enc4  : DS2C block                     → (1024, H/16, W/16)  ← + LightAttn
  → ADASPP (rates 3,6,12,18,24)            → (1024, H/16, W/16)
  → Dec3  : up + skip + Conv               →  (512,  H/8,  W/8)
  → Dec2  : up + skip + Conv               →  (256,  H/4,  W/4)
  → Dec1  : up + skip + Conv               →  (128,  H/2,  W/2)
  → Dec0  : up + skip + Conv               →   (64,   H,    W)
  → Head  : Conv1×1                        → (n_cls,  H,    W)

Key modules
───────────
  DS2C       — Depth-wise Separable Asymmetric Conv block (§2.1.1–2.1.2)
               Parallel: 3×3 DSC | 1×3 conv | 3×1 conv → sum → BN → ReLU
               + inverse-bottleneck + residual connection

  LightAttn  — Lightweight Asymmetric Self-Attention (§2.1.3, Eqs. 13–16)
               Q: full-resolution (H·W)×C
               K,V: spatially reduced (H/r·W/r)×C via stride-r conv

  ADASPP     — Adaptive Atrous Spatial Pyramid Pooling (§2.2, Fig. 3)
               5 dilated branches (rates 3,6,12,18,24)
               Cumulative inputs: branch_i sees encoder + Σ prev branch outputs
               Each branch output scaled by learnable weight (init=1)
               Final: concat → 1×1 conv

Usage
─────
  from c2s_roadnet import C2SRoadNet

  model = C2SRoadNet(n_classes=1).to(device)
  logits = model(imgs)      # (B, 1, H, W) — raw logits; apply sigmoid / BCEWithLogitsLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  Depth-wise Separable Asymmetric Convolution Block  (DS2C)
# ─────────────────────────────────────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution: depthwise 3×3 → pointwise 1×1.
    Replaces the 3×3 branch in the asymmetric convolution block.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                            padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class DS2C(nn.Module):
    """
    Depth-wise Separable Asymmetric Convolution block (§2.1.1–2.1.2, Fig. 1–2).

    Eq. 6:  X̄_i = F_DSC(X_{i-1}) + F_{1×3}(X_{i-1}) + F_{3×1}(X_{i-1})
    Eq. 7:  X_i  = BN(X̄_i)   → activation

    Wrapped in an inverse-bottleneck residual:
        expand (1×1) → DS2C parallel branches → project (1×1) → + shortcut

    Args:
        in_ch    : input channels
        out_ch   : output channels
        stride   : spatial stride (1 or 2 for downsampling)
        expand   : expansion ratio for the inverse-bottleneck mid channels
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand: int = 4):
        super().__init__()
        mid_ch = in_ch * expand

        # Expand
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
        )

        # Parallel asymmetric branches (operate on mid_ch channels)
        self.branch_dsc = DepthwiseSeparableConv(mid_ch, mid_ch, stride=stride)  # 3×3 DSC
        self.branch_1x3 = nn.Conv2d(mid_ch, mid_ch, (1, 3),
                                    stride=stride, padding=(0, 1), bias=False)
        self.branch_3x1 = nn.Conv2d(mid_ch, mid_ch, (3, 1),
                                    stride=stride, padding=(1, 0), bias=False)
        self.bn_merge   = nn.BatchNorm2d(mid_ch)
        self.act        = nn.ReLU6(inplace=True)

        # Project back
        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        # Shortcut: match channels and stride
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.final_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        # Inverse bottleneck expand
        h = self.expand(x)

        # Parallel asymmetric branches (Eq. 6)
        h = self.act(self.bn_merge(
            self.branch_dsc(h) + self.branch_1x3(h) + self.branch_3x1(h)
        ))

        # Project (Eq. 7)
        h = self.project(h)

        return self.final_act(h + identity)


# ─────────────────────────────────────────────────────────────────────────────
#  Lightweight Asymmetric Self-Attention  (LightAttn)
# ─────────────────────────────────────────────────────────────────────────────

class LightweightAsymmetricAttention(nn.Module):
    """
    Lightweight asymmetric multi-head self-attention (§2.1.3, Eqs. 13–16).

    "Asymmetric" means Q operates at full spatial resolution while K and V
    operate at a spatially reduced resolution (factor r), giving:
        Q shape :  (B, N,  C)   where N  = H × W
        K,V shape: (B, Nℵ, C)   where Nℵ = (H/r) × (W/r)

    Eq. 16:  Attn(Q,K,V) = softmax( Q Kℵ^T / √dk + B ) Vℵ

    Args:
        channels   : feature channels (must be divisible by num_heads)
        num_heads  : number of attention heads
        reduction  : spatial reduction factor r for K and V
    """

    def __init__(self, channels: int, num_heads: int = 8, reduction: int = 2):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.dk        = channels // num_heads
        self.scale     = self.dk ** -0.5

        # Q: 1×1 conv at full resolution
        self.to_q = nn.Conv2d(channels, channels, 1, bias=False)

        # K, V: stride-r conv → reduced spatial resolution
        self.to_k = nn.Conv2d(channels, channels, reduction,
                              stride=reduction, bias=False)
        self.to_v = nn.Conv2d(channels, channels, reduction,
                              stride=reduction, bias=False)

        # Learnable bias B (per head, per (Q-position, K-position) pair is too large;
        # we use a lightweight positional scalar bias per head instead)
        self.bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        nh, dk = self.num_heads, self.dk

        # Compute Q (full res), K and V (reduced res)
        Q = self.to_q(x)                           # (B, C, H,   W)
        K = self.to_k(x)                           # (B, C, H/r, W/r)
        V = self.to_v(x)

        Hr, Wr = K.shape[2], K.shape[3]
        N  = H * W
        Nr = Hr * Wr

        # Reshape to (B, nh, seq, dk)
        def reshape(t, seq):
            return t.reshape(B, nh, dk, seq).transpose(2, 3)  # (B, nh, seq, dk)

        Q = reshape(Q.reshape(B, C, N), N)          # (B, nh, N,  dk)
        K = reshape(K.reshape(B, C, Nr), Nr)        # (B, nh, Nr, dk)
        V = reshape(V.reshape(B, C, Nr), Nr)

        # Scaled dot-product attention with additive bias B (Eq. 16)
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, nh, N, Nr)
        attn = (attn + self.bias).softmax(dim=-1)

        out = attn @ V                              # (B, nh, N, dk)
        out = out.transpose(2, 3).reshape(B, C, H, W)

        # Residual + norm
        out = x + self.proj(out)
        # LayerNorm on channel dim
        out = self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-Scale Adaptive Weight Module  (ADASPP)
# ─────────────────────────────────────────────────────────────────────────────

class ADASPP(nn.Module):
    """
    Adaptive Atrous Spatial Pyramid Pooling (§2.2, Fig. 3).

    5 dilated branches with rates [3, 6, 12, 18, 24].
    Cumulative input scheme (from paper §2.3):
        branch_3  input = encoder_out
        branch_6  input = encoder_out + branch_3_out
        branch_12 input = encoder_out + branch_3_out + branch_6_out
        branch_18 input = encoder_out + ... + branch_12_out
        branch_24 input = encoder_out + ... + branch_18_out

    Each branch output is multiplied by a learnable scalar weight w_i (init=1).
    Final: concat of all 5 weighted outputs → 1×1 BN+ReLU conv.

    Args:
        in_ch  : input channels from encoder
        out_ch : output channels
        rates  : dilation rates for the 5 branches
    """

    def __init__(
        self,
        in_ch:  int,
        out_ch: int,
        rates:  list = None,
    ):
        super().__init__()
        if rates is None:
            rates = [3, 6, 12, 18, 24]
        n = len(rates)

        # Each branch: same in_ch → in_ch so cumulative additions stay at in_ch
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3,
                          padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
            )
            for r in rates
        ])

        # Learnable scalar weights w1..w5 (init=1, Eq. from §2.2)
        self.weights = nn.Parameter(torch.ones(n))

        # Final 1×1 conv: collapse n*in_ch → out_ch
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch * n, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, in_ch, H, W) — last encoder feature map
        Returns:
            fused : (B, out_ch, H, W)
        """
        outputs    = []
        cumulative = x    # starts as encoder output

        for i, branch in enumerate(self.branches):
            out        = branch(cumulative)                  # dilated conv
            outputs.append(out * self.weights[i])            # learnable scale
            cumulative = cumulative + out                     # accumulate for next branch

        return self.fuse(torch.cat(outputs, dim=1))


# ─────────────────────────────────────────────────────────────────────────────
#  Encoder stage helpers
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two standard 3×3 Conv + BN + ReLU layers (used for Enc0 and decoder)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    """Upsample ×2 → concat skip → ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle any 1-pixel size mismatch from odd-sized inputs
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
#  Full C2S-RoadNet
# ─────────────────────────────────────────────────────────────────────────────

class C2SRoadNet(nn.Module):
    """
    C2S-RoadNet: Road extraction via DS2C + Lightweight Asymmetric Self-Attention
    + Adaptive ASPP (Yin et al., Remote Sensing 2023).

    Spatial flow (256×256 example)
    ───────────────────────────────
      Enc0   : 3  → 64,  H/2  (Conv3×3×2 + Pool)
      Enc1   : 64 → 128, H/4  (DS2C × 2 + Pool)
      Enc2   : 128→ 256, H/8  (DS2C × 2 + Pool)
      Enc3   : 256→ 512, H/16 (DS2C × 2 + Pool)
      Enc4   : 512→1024, H/16 (DS2C × 2, no pool → bottleneck)
      LightAttn at bottleneck
      ADASPP : 1024→1024, H/16
      Dec3   : 1024 + 512 → 512, H/8
      Dec2   : 512  + 256 → 256, H/4
      Dec1   : 256  + 128 → 128, H/2
      Dec0   : 128  +  64 →  64, H
      Head   : 64  → n_classes, H

    Args:
        n_classes   : 1 for binary road segmentation (raw logits → sigmoid)
        base_ch     : base channel count (default 64)
        attn_heads  : number of attention heads in LightAttn
        attn_reduce : spatial reduction factor r in LightAttn (K, V)
        adaspp_rates: dilation rates for the 5 ADASPP branches
    """

    def __init__(
        self,
        n_classes:    int  = 1,
        base_ch:      int  = 64,
        attn_heads:   int  = 8,
        attn_reduce:  int  = 2,
        adaspp_rates: list = None,
    ):
        super().__init__()
        if adaspp_rates is None:
            adaspp_rates = [3, 6, 12, 18, 24]

        c = base_ch   # 64
        ch = [c, c * 2, c * 4, c * 8, c * 16]   # [64, 128, 256, 512, 1024]

        # ── Encoder ──────────────────────────────────────────────────────────
        # Stage 0: standard double-conv (paper: first stage uses regular Conv)
        self.enc0     = ConvBlock(3, ch[0])
        self.pool0    = nn.MaxPool2d(2)

        # Stages 1–3: two DS2C blocks each + maxpool
        self.enc1     = nn.Sequential(DS2C(ch[0], ch[1]), DS2C(ch[1], ch[1]))
        self.pool1    = nn.MaxPool2d(2)

        self.enc2     = nn.Sequential(DS2C(ch[1], ch[2]), DS2C(ch[2], ch[2]))
        self.pool2    = nn.MaxPool2d(2)

        self.enc3     = nn.Sequential(DS2C(ch[2], ch[3]), DS2C(ch[3], ch[3]))
        self.pool3    = nn.MaxPool2d(2)

        # Stage 4 (bottleneck, no pool): two DS2C + lightweight self-attention
        self.enc4     = nn.Sequential(DS2C(ch[3], ch[4]), DS2C(ch[4], ch[4]))
        self.attn     = LightweightAsymmetricAttention(ch[4], attn_heads, attn_reduce)

        # ── ADASPP ───────────────────────────────────────────────────────────
        self.adaspp   = ADASPP(ch[4], ch[4], rates=adaspp_rates)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec3     = DecoderBlock(ch[4], ch[3], ch[3])   # 1024+512 → 512
        self.dec2     = DecoderBlock(ch[3], ch[2], ch[2])   # 512+256  → 256
        self.dec1     = DecoderBlock(ch[2], ch[1], ch[1])   # 256+128  → 128
        self.dec0     = DecoderBlock(ch[1], ch[0], ch[0])   # 128+64   → 64

        # ── Segmentation head ─────────────────────────────────────────────────
        self.head     = nn.Conv2d(ch[0], n_classes, 1)

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, H, W)

        Returns:
            logits : (B, n_classes, H, W)  — raw, no sigmoid/softmax
        """
        # ── Encoder ──────────────────────────────────────────────────────────
        s0 = self.enc0(x)                   # (B,   64, H/2,  W/2)   ← skip
        s1 = self.enc1(self.pool0(s0))      # (B,  128, H/4,  W/4)   ← skip
        s2 = self.enc2(self.pool1(s1))      # (B,  256, H/8,  W/8)   ← skip
        s3 = self.enc3(self.pool2(s2))      # (B,  512, H/16, W/16)  ← skip
        e4 = self.enc4(self.pool3(s3))      # (B, 1024, H/16, W/16)

        # ── Self-attention + ADASPP at bottleneck ────────────────────────────
        e4 = self.attn(e4)                  # lightweight asymmetric attention
        e4 = self.adaspp(e4)                # adaptive ASPP fusion

        # ── Decoder ──────────────────────────────────────────────────────────
        d  = self.dec3(e4, s3)              # (B, 512, H/8,  W/8)
        d  = self.dec2(d,  s2)             # (B, 256, H/4,  W/4)
        d  = self.dec1(d,  s1)             # (B, 128, H/2,  W/2)
        d  = self.dec0(d,  s0)             # (B,  64, H,    W)

        return self.head(d)                 # (B, n_classes, H, W)


# ─────────────────────────────────────────────────────────────────────────────
#  Sanity check  (python c2s_roadnet.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}\n')

    model = C2SRoadNet(n_classes=1).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_params / 1e6:.2f} M\n')

    # Module-level parameter breakdown
    for name, module in [
        ('Enc0',    model.enc0),
        ('Enc1',    model.enc1),
        ('Enc2',    model.enc2),
        ('Enc3',    model.enc3),
        ('Enc4',    model.enc4),
        ('LightAttn', model.attn),
        ('ADASPP',  model.adaspp),
        ('Dec3',    model.dec3),
        ('Dec2',    model.dec2),
        ('Dec1',    model.dec1),
        ('Dec0',    model.dec0),
        ('Head',    model.head),
    ]:
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f'  {name:<12} {n / 1e6:>6.2f} M')

    print()
    for H, W in [(256, 256), (512, 512)]:
        dummy  = torch.randn(2, 3, H, W, device=device)
        logits = model(dummy)
        assert logits.shape == (2, 1, H, W), f'Shape mismatch: {logits.shape}'
        print(f'Input {tuple(dummy.shape)}  →  Output {tuple(logits.shape)}  ✓')

    # Verify ADASPP learnable weights
    w = model.adaspp.weights.detach().cpu()
    print(f'\nADASPP initial weights: {w.tolist()}  (should all be 1.0)')

    # Verify LightAttn bias
    b = model.attn.bias.detach().cpu()
    print(f'LightAttn bias shape: {tuple(b.shape)}  (should be (1, 8, 1, 1))')

    print('\nAll checks passed.')