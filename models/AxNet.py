"""
OurNet — Road Extraction via Channel Attention and Multilayer Axial Transformer
PyTorch implementation based on:

  Meng et al., "Road Extraction From Remote Sensing Images via
  Channel Attention and Multilayer Axial Transformer"
  IEEE Geoscience and Remote Sensing Letters, Vol. 21, 2024.
  DOI: 10.1109/LGRS.2024.3379502

Architecture (Fig. 1 in paper)
───────────────────────────────
Input (3 ch, H×W)
  → ResNet-34 encoder (5 stages, strides ×2 at stages 0,2,3,4)
      Stage 0 : conv7×7 + BN + ReLU + MaxPool  →  (64,  H/4,  W/4)
      Stage 1 : layer1 (3×BasicBlock, stride 1) →  (64,  H/4,  W/4)
      Stage 2 : layer2 (4×BasicBlock, stride 2) → (128,  H/8,  W/8)
      Stage 3 : layer3 (6×BasicBlock, stride 2) → (256, H/16, W/16)
      Stage 4 : layer4 (3×BasicBlock, stride 2) → (512, H/32, W/32)
  → CAM on each encoder output (for skip connections)
  → 3 stacked ATMs at bottleneck            → (512, H/32, W/32) × 3
  → MLAF fuses 3 ATM outputs               → (512, H/32, W/32)
  → 4 decoder blocks (transposed conv + skip + conv3 + conv1)
  → Final head (transposed conv + conv1×1)  → (n_classes, H, W)

Key modules
───────────
  CAM  — Channel Attention Module (Eq. 1, Fig. 2)
  ATM  — Axial Transformer Module: height-axis + width-axis attention (Fig. 3)
  MLAF — Multilayer Attention Fusion Module (Fig. 4)

Usage
─────
  from ournet import OurNet

  model = OurNet(n_classes=1, num_heads=8, pretrained=True).to(device)
  logits = model(imgs)          # (B, 1, H, W) — raw logits, no sigmoid
  # loss: BCEWithLogitsLoss  or  DiceLoss + BCE  (as in the paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


# ─────────────────────────────────────────────────────────────────────────────
#  Channel Attention Module  (CAM)
# ─────────────────────────────────────────────────────────────────────────────

class CAM(nn.Module):
    """
    Channel Attention Module — applied on every encoder skip connection.

    Eq. 1:  Fout = σ(ω ⊗ BN(W1×1(W3×3(GAP(Fin))))) ⊗ Fin
    where   ω_i = c_i / Σ_i c_i   (channel-wise scale factor)

    Args:
        ch : number of input/output channels
    """

    def __init__(self, ch: int):
        super().__init__()
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.conv3 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)  # W3×3
        self.conv1 = nn.Conv2d(ch, ch, 1, bias=False)             # W1×1
        self.bn    = nn.BatchNorm2d(ch)
        self.sig   = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, 1, 1) — squeeze spatial dims via GAP, then project
        c = self.bn(self.conv1(self.conv3(self.gap(x))))
        # ω_i = c_i / Σ c_i  (normalize over channels)
        omega = c / c.sum(dim=1, keepdim=True).clamp(min=1e-6)
        # channel attention weights
        w = self.sig(omega * c)                                    # (B, C, 1, 1)
        return x * w


# ─────────────────────────────────────────────────────────────────────────────
#  Single-axis multi-head self-attention  (building block of ATM)
# ─────────────────────────────────────────────────────────────────────────────

class AxisAttention(nn.Module):
    """
    Multi-head self-attention along ONE spatial axis (height or width).

    Complexity is O(max(H, W)) rather than O(H²W²) for standard 2-D attention.
    For the height axis:  each of the W columns gets an independent H×H attention.
    For the width  axis:  each of the H rows  gets an independent W×W attention.

    Args:
        channels  : feature channels (must be divisible by num_heads)
        num_heads : number of attention heads
        axis      : 'height' or 'width'
    """

    def __init__(self, channels: int, num_heads: int = 8, axis: str = 'height'):
        super().__init__()
        assert axis in ('height', 'width'), "axis must be 'height' or 'width'"
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.axis      = axis
        self.num_heads = num_heads
        self.dk        = channels // num_heads
        self.scale     = self.dk ** -0.5
        self.to_qkv    = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj      = nn.Conv2d(channels, channels,     1, bias=False)

    # ── helper: rearrange (B, C, H, W) → (B*seq_outer, nh, seq_inner, dk) ──

    def _prep_height(self, t: torch.Tensor, B, C, H, W):
        """(B, C, H, W) → (B*W, nh, H, dk)  for height-axis attention."""
        nh, dk = self.num_heads, self.dk
        t = t.reshape(B, nh, dk, H, W)     # (B, nh, dk, H, W)
        t = t.permute(0, 4, 1, 3, 2)       # (B, W, nh, H, dk)
        return t.reshape(B * W, nh, H, dk)

    def _unprep_height(self, t: torch.Tensor, B, C, H, W):
        """(B*W, nh, H, dk) → (B, C, H, W)."""
        nh, dk = self.num_heads, self.dk
        t = t.reshape(B, W, nh, H, dk)     # (B, W, nh, H, dk)
        t = t.permute(0, 2, 4, 3, 1)       # (B, nh, dk, H, W)
        return t.reshape(B, C, H, W)

    def _prep_width(self, t: torch.Tensor, B, C, H, W):
        """(B, C, H, W) → (B*H, nh, W, dk)  for width-axis attention."""
        nh, dk = self.num_heads, self.dk
        t = t.reshape(B, nh, dk, H, W)     # (B, nh, dk, H, W)
        t = t.permute(0, 3, 1, 4, 2)       # (B, H, nh, W, dk)
        return t.reshape(B * H, nh, W, dk)

    def _unprep_width(self, t: torch.Tensor, B, C, H, W):
        """(B*H, nh, W, dk) → (B, C, H, W)."""
        nh, dk = self.num_heads, self.dk
        t = t.reshape(B, H, nh, W, dk)     # (B, H, nh, W, dk)
        t = t.permute(0, 2, 4, 1, 3)       # (B, nh, dk, H, W)
        return t.reshape(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)   # each (B, C, H, W)

        prep   = self._prep_height   if self.axis == 'height' else self._prep_width
        unprep = self._unprep_height if self.axis == 'height' else self._unprep_width

        Q = prep(q, B, C, H, W)   # (B*outer, nh, inner, dk)
        K = prep(k, B, C, H, W)
        V = prep(v, B, C, H, W)

        # Scaled dot-product attention along the chosen axis
        attn = (Q @ K.transpose(-2, -1)) * self.scale   # (..., inner, inner)
        attn = attn.softmax(dim=-1)
        out  = attn @ V                                  # (..., inner, dk)

        return self.proj(unprep(out, B, C, H, W))


# ─────────────────────────────────────────────────────────────────────────────
#  Axial Transformer Module  (ATM)
# ─────────────────────────────────────────────────────────────────────────────

class ATM(nn.Module):
    """
    Axial Transformer Module (Fig. 1, Fig. 3).

    Structure:
        LN → height-axis attention → residual
        LN → width-axis  attention → residual
        LN → position-wise FFN     → residual

    Args:
        channels  : feature channels
        num_heads : number of attention heads
        mlp_ratio : FFN hidden dimension multiplier
    """

    def __init__(self, channels: int, num_heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.ln_h  = nn.LayerNorm(channels)
        self.ln_w  = nn.LayerNorm(channels)
        self.ln_ff = nn.LayerNorm(channels)
        self.h_attn = AxisAttention(channels, num_heads, axis='height')
        self.w_attn = AxisAttention(channels, num_heads, axis='width')
        hidden = channels * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    # ── LayerNorm wrapper for (B, C, H, W) tensors ────────────────────────

    @staticmethod
    def _ln(ln: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        """Apply LN on channel dim: (B,C,H,W) → permute → LN → permute back."""
        return ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Height-axis attention
        x = x + self.h_attn(self._ln(self.ln_h, x))
        # Width-axis attention
        x = x + self.w_attn(self._ln(self.ln_w, x))
        # FFN (applied per-pixel in channel space)
        B, C, H, W = x.shape
        xp = x.permute(0, 2, 3, 1)                # (B, H, W, C)
        xp = xp + self.ffn(self.ln_ff(xp))
        return xp.permute(0, 3, 1, 2)             # (B, C, H, W)


# ─────────────────────────────────────────────────────────────────────────────
#  Multilayer Attention Fusion Module  (MLAF)
# ─────────────────────────────────────────────────────────────────────────────

class MLAF(nn.Module):
    """
    Multilayer Attention Fusion Module (Fig. 4).

    Treats the three ATM output feature maps as three "tokens",
    each of dimensionality C×H×W, and computes a 3×3 cross-token
    correlation attention matrix to produce a single fused output.

    Steps (as described in §II-D):
        1. Stack feat_list → Q̂ : (B, 3, C·H·W)
        2. K̂ = Q̂ᵀ           : (B, C·H·W, 3)
        3. Correlation  = Q̂ @ K̂ / α → softmax → (B, 3, 3)
        4. V̂  = K̂             : (B, C·H·W, 3)
        5. Out = V̂ @ Corrᵀ    : (B, C·H·W, 3) → reshape (B, 3C, H, W)
        6. 1×1 conv            : (B, 3C, H, W) → (B, C, H, W)

    Args:
        channels : feature channels C (per ATM output)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv_out = nn.Conv2d(channels * 3, channels, 1, bias=False)
        self.bn       = nn.BatchNorm2d(channels)
        self.relu     = nn.ReLU(inplace=True)

    def forward(self, feat_list: list) -> torch.Tensor:
        """
        Args:
            feat_list : [f1, f2, f3], each (B, C, H, W)
        Returns:
            fused     : (B, C, H, W)
        """
        B, C, H, W = feat_list[0].shape

        # Q̂: (B, 3, C*H*W)  — each feature map as one "token"
        tokens = torch.stack([f.flatten(2) for f in feat_list], dim=1)

        Q = tokens                      # (B, 3,   CHW)
        K = tokens.transpose(1, 2)      # (B, CHW, 3)
        V = K                           # (B, CHW, 3)

        # 3×3 correlation attention
        alpha = (C * H * W) ** 0.5
        corr  = torch.bmm(Q, K).div(alpha).softmax(dim=-1)   # (B, 3, 3)

        # Weighted value aggregation: (B, CHW, 3) @ (B, 3, 3) = (B, CHW, 3)
        out = torch.bmm(V, corr)                              # (B, CHW, 3)

        # (B, CHW, 3) → (B, 3, CHW) → (B, 3C, H, W) → 1×1 conv → (B, C, H, W)
        out = out.transpose(1, 2).reshape(B, 3 * C, H, W)
        return self.relu(self.bn(self.conv_out(out)))


# ─────────────────────────────────────────────────────────────────────────────
#  Decoder Block
# ─────────────────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Single decoder stage: upsample → cat with skip → Conv3×3 → Conv1×1.
    Mirrors the "Decoder Block" in Fig. 1 (transposed-conv + conv3 + conv1).

    Args:
        in_ch   : channels from the lower (deeper) decoder stage
        skip_ch : channels from the CAM-processed encoder skip
        out_ch  : output channels
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # Transposed conv: upsample ×2 and reduce channels to out_ch
        self.up    = nn.ConvTranspose2d(in_ch, out_ch, 3,
                                        stride=2, padding=1, output_padding=1)
        # After cat with skip: (out_ch + skip_ch) channels
        self.conv3 = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)                          # upsample ×2
        x = torch.cat([x, skip], dim=1)         # fuse with encoder skip
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn1(self.conv1(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  Full OurNet
# ─────────────────────────────────────────────────────────────────────────────

class axnet(nn.Module):
    """
    Road extraction model with Channel Attention and Multilayer Axial Transformer.

    Meng et al., IEEE GRSL 2024.

    Spatial resolution (example for 256×256 input)
    ─────────────────────────────────────────────
      Input          :  (B,   3, 256, 256)
      enc0 (conv+LN) :  (B,  64, 128, 128)   ← stride 2
      enc0 + maxpool :  (B,  64,  64,  64)   ← stride 4
      enc1 (layer1)  :  (B,  64,  64,  64)   stride 1
      enc2 (layer2)  :  (B, 128,  32,  32)   ← stride 8
      enc3 (layer3)  :  (B, 256,  16,  16)   ← stride 16
      enc4 (layer4)  :  (B, 512,   8,   8)   ← stride 32  ← ATMs here
      MLAF           :  (B, 512,   8,   8)
      dec4 + s3      :  (B, 256,  16,  16)
      dec3 + s2      :  (B, 128,  32,  32)
      dec2 + s1      :  (B,  64,  64,  64)
      dec1 + s0      :  (B,  64, 128, 128)
      head           :  (B, n_classes, 256, 256)

    Args:
        n_classes  : output channels (1 for binary road segmentation)
        num_heads  : attention heads in each ATM
        pretrained : load ImageNet-pretrained ResNet-34 weights
    """

    def __init__(
        self,
        n_classes:  int  = 1,
        num_heads:  int  = 8,
        pretrained: bool = True,
    ):
        super().__init__()

        # ── Encoder: ResNet-34 ────────────────────────────────────────────────
        weights  = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)

        # Stage 0: 7×7 conv, BN, ReLU  (stride 2)
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool   # stride 2  (cumulative stride = 4)

        self.enc1 = backbone.layer1    # 3× BasicBlock, 64  ch, stride 1
        self.enc2 = backbone.layer2    # 4× BasicBlock, 128 ch, stride 2
        self.enc3 = backbone.layer3    # 6× BasicBlock, 256 ch, stride 2
        self.enc4 = backbone.layer4    # 3× BasicBlock, 512 ch, stride 2

        # ── CAM on skip connections ───────────────────────────────────────────
        self.cam0 = CAM(64)            # s0: 64  ch, H/2  × W/2
        self.cam1 = CAM(64)            # s1: 64  ch, H/4  × W/4
        self.cam2 = CAM(128)           # s2: 128 ch, H/8  × W/8
        self.cam3 = CAM(256)           # s3: 256 ch, H/16 × W/16

        # ── Three stacked ATMs at bottleneck (512 ch) ────────────────────────
        self.atm1 = ATM(512, num_heads)
        self.atm2 = ATM(512, num_heads)
        self.atm3 = ATM(512, num_heads)

        # ── MLAF: fuse 3 ATM outputs → single bottleneck ─────────────────────
        self.mlaf = MLAF(512)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Each DecoderBlock(in_ch, skip_ch, out_ch)
        self.dec4 = DecoderBlock(512, 256, 256)   # → H/16, 256 ch
        self.dec3 = DecoderBlock(256, 128, 128)   # → H/8,  128 ch
        self.dec2 = DecoderBlock(128,  64,  64)   # → H/4,   64 ch
        self.dec1 = DecoderBlock( 64,  64,  64)   # → H/2,   64 ch

        # ── Segmentation head ─────────────────────────────────────────────────
        # Final upsample ×2 → H, then 1×1 conv → n_classes
        self.head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, 1),
        )

        # Initialise non-pretrained weights
        self._init_weights()

    # ── Weight init ───────────────────────────────────────────────────────────

    def _init_weights(self):
        """Kaiming normal for all Conv2d layers outside the backbone."""
        skip_ids = set(id(p)
                       for m in [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]
                       for p in m.parameters())
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if id(m.weight) not in skip_ids:
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

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, H, W)  — RGB remote sensing image patch

        Returns:
            logits : (B, n_classes, H, W)  — raw logits (no sigmoid)
                     use BCEWithLogitsLoss  (n_classes=1)
                     or CrossEntropyLoss   (n_classes=2)
        """
        # ── Encoder ──────────────────────────────────────────────────────────
        e0 = self.enc0(x)           # (B,  64, H/2,  W/2)
        e1 = self.enc1(self.pool(e0))  # (B,  64, H/4,  W/4)   pool → layer1
        e2 = self.enc2(e1)          # (B, 128, H/8,  W/8)
        e3 = self.enc3(e2)          # (B, 256, H/16, W/16)
        e4 = self.enc4(e3)          # (B, 512, H/32, W/32)

        # ── CAM-weighted skip connections ─────────────────────────────────────
        s0 = self.cam0(e0)          # (B,  64, H/2,  W/2)
        s1 = self.cam1(e1)          # (B,  64, H/4,  W/4)
        s2 = self.cam2(e2)          # (B, 128, H/8,  W/8)
        s3 = self.cam3(e3)          # (B, 256, H/16, W/16)

        # ── Axial Transformer bottleneck ──────────────────────────────────────
        a1 = self.atm1(e4)          # (B, 512, H/32, W/32)
        a2 = self.atm2(a1)
        a3 = self.atm3(a2)

        # ── Multilayer fusion ─────────────────────────────────────────────────
        bottleneck = self.mlaf([a1, a2, a3])   # (B, 512, H/32, W/32)

        # ── Decoder ──────────────────────────────────────────────────────────
        d = self.dec4(bottleneck, s3)    # (B, 256, H/16, W/16)
        d = self.dec3(d, s2)             # (B, 128, H/8,  W/8)
        d = self.dec2(d, s1)             # (B,  64, H/4,  W/4)
        d = self.dec1(d, s0)             # (B,  64, H/2,  W/2)

        return self.head(d)              # (B, n_classes, H, W)


# ─────────────────────────────────────────────────────────────────────────────
#  Quick sanity check  (python ournet.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}\n')

    model = axnet(n_classes=1, num_heads=8, pretrained=False).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params : {n_params / 1e6:.2f} M  '
          f'(paper reports ~47.12 M with pretrained backbone)')

    # Test with 256×256 and 1024×1024 inputs
    for H, W in [(256, 256), (512, 512)]:
        dummy  = torch.randn(2, 3, H, W, device=device)
        logits = model(dummy)
        assert logits.shape == (2, 1, H, W), f'Shape mismatch: {logits.shape}'
        print(f'Input {tuple(dummy.shape)}  →  Output {tuple(logits.shape)}  ✓')

    print('\nAll checks passed.')