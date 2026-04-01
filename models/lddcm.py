"""
L-DDCM-Net : Light Dense Dilated Convolutions Merging Network
PyTorch implementation based on:

  Georges et al., "Automatic forest road extraction from LiDAR data
  using convolutional neural networks" (2022)

  Original TF code : https://github.com/paulgeorges1998/Light-DDCM-Net

Architecture (Fig. 5 in paper)
──────────────────────────────
Input (N ch)
  → channel adapter           → (B,  3,    H,    W)
  → ResNet50 layer1–3         → (B, 1024, H/16, W/16)   stride-16 features
  → DDCM [ rates 1,2,3,4 ]   → (B,  36,  H/16, W/16)
  → bilinear ×4               → (B,  36,  H/4,  W/4 )
  → DDCM [ rate 1 ]           → (B,  18,  H/4,  W/4 )
  → bilinear ×2               → (B,  18,  H/2,  W/2 )
  → 1×1 conv                  → (B,   1,  H/2,  W/2 )   (logits)
  → bilinear ×2               → (B,   1,  H,    W   )   (full resolution)

Spatial math check (input 256×256)
───────────────────────────────────
  ResNet up to layer3  →  16×16
  ×4 upsample          →  64×64
  ×2 upsample          → 128×128
  ×2 upsample          → 256×256  ✓

Drop-in usage
─────────────
  from models.lddcm import LDDCM_Net

  model = LDDCM_Net(n_channels=4, n_classes=1).to(device)
  logits = model(imgs)          # (B, 1, H, W) — no sigmoid inside
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# ─────────────────────────────────────────────────────────────────────────────
#  DDCM block  (Fig. 4 in paper)
# ─────────────────────────────────────────────────────────────────────────────

class DDCM_Block(nn.Module):
    """
    Dense Dilated Convolutions Merging block.

    For each dilation rate k:
      feat = DilatedConv(rate=k) → PReLU → BN          ← DC block
      x    = concat(x, feat)                            ← dense skip
    
    Then a 1×1 Merging block collapses the accumulated channels back to
    out_channels.

    Channel growth inside the block:
      after rate[0]: in_ch + out_ch
      after rate[1]: in_ch + 2*out_ch
      ...
      after rate[n]: in_ch + (n+1)*out_ch  → fed into merge conv

    Args:
        in_channels  : channels arriving at this block
        out_channels : channels produced by every DC sub-block (and the final merge)
        rates        : list of dilation rates, e.g. [1, 2, 3, 4]
    """

    def __init__(self, in_channels: int, out_channels: int, rates: list):
        super().__init__()

        self.dc_blocks = nn.ModuleList()
        current_ch = in_channels

        for rate in rates:
            # Each DC block: dilated conv → PReLU → BN
            # (Paper's diagram shows PReLU+BN; the TF code applies relu-in-conv
            #  then PReLU then BN — we follow the diagram and use Conv → PReLU → BN)
            self.dc_blocks.append(nn.Sequential(
                nn.Conv2d(current_ch, out_channels, kernel_size=3,
                          padding=rate, dilation=rate, bias=False),
                nn.PReLU(out_channels),
                nn.BatchNorm2d(out_channels),
            ))
            current_ch += out_channels      # dense concatenation grows channels

        # Merging block: 1×1 conv collapses all stacked features
        self.merge = nn.Sequential(
            nn.Conv2d(current_ch, out_channels, kernel_size=1, bias=False),
            nn.PReLU(out_channels),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for dc in self.dc_blocks:
            feat = dc(x)
            x = torch.cat([x, feat], dim=1)     # dense concatenation
        return self.merge(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Optional CBAM blocks  (Section 4 in paper — L-DDCM+3-CBAM / +8-CBAM)
# ─────────────────────────────────────────────────────────────────────────────

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // ratio, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.shared_mlp(self.avg_pool(x))
        mx  = self.shared_mlp(self.max_pool(x))
        return x * self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        att = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018)."""
    def __init__(self, channels: int, ratio: int = 8, kernel_size: int = 3):
        super().__init__()
        self.channel = ChannelAttention(channels, ratio)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial(self.channel(x))


# ─────────────────────────────────────────────────────────────────────────────
#  Full L-DDCM-Net
# ─────────────────────────────────────────────────────────────────────────────

class LDDCM_Net(nn.Module):
    """
    Light DDCM-Net for binary segmentation of forest roads from LiDAR DTM.

    Args:
        n_channels  : input image channels  (1=grayscale, 4=hillshade+extras)
        n_classes   : output channels       (1 for binary segmentation)
        use_cbam    : if True, adds a CBAM module after the ResNet encoder and
                      after each DDCM block (matches L-DDCM+3-CBAM variant)
        pretrained  : load ImageNet weights for ResNet50 backbone
    
    Forward output:
        Raw logits  (B, n_classes, H, W)  — apply sigmoid / BCEWithLogitsLoss
    """

    def __init__(
        self,
        n_channels: int = 1,
        n_classes:  int = 1,
        use_cbam:   bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        self.use_cbam = use_cbam

        # ── 1. Channel adapter ────────────────────────────────────────────────
        # Paper (N_CHANNELS=1): conv(1→2) + concat-with-input → 3ch → BN+PReLU.
        # For arbitrary N_CHANNELS we use a learned 3×3 conv projection to 3ch
        # followed by BN+PReLU — same spirit, generalises cleanly to 4 channels.
        self.channel_adapter = nn.Sequential(
            nn.Conv2d(n_channels, 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.PReLU(3),
        )

        # ── 2. ResNet50 encoder (layers 1–3 only) ─────────────────────────────
        # "Remove the last bottleneck layer" (paper §2.3) = drop layer4.
        # Spatial stride after layer3 = 16, so a 256×256 input → 16×16 features.
        # Output channels = 1024.
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)

        self.encoder = nn.Sequential(
            backbone.conv1,      # 7×7, stride 2
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,    # stride 2   → cumulative stride 4
            backbone.layer1,     # 3× bottleneck,  256 ch, cumulative stride 4
            backbone.layer2,     # 4× bottleneck,  512 ch, cumulative stride 8
            backbone.layer3,     # 6× bottleneck, 1024 ch, cumulative stride 16
            # layer4 intentionally excluded
        )
        resnet_out_ch = 1024

        # Optional CBAM after encoder (L-DDCM+3-CBAM variant)
        self.cbam_enc   = CBAM(resnet_out_ch)  if use_cbam else nn.Identity()

        # ── 3. DDCM decoder blocks ────────────────────────────────────────────
        ddcm1_ch = 36
        ddcm2_ch = 18

        self.ddcm1     = DDCM_Block(resnet_out_ch, ddcm1_ch, rates=[1, 2, 3, 4])
        self.cbam_d1   = CBAM(ddcm1_ch)        if use_cbam else nn.Identity()

        self.ddcm2     = DDCM_Block(ddcm1_ch,    ddcm2_ch, rates=[1])
        self.cbam_d2   = CBAM(ddcm2_ch)        if use_cbam else nn.Identity()

        # ── 4. Segmentation head ──────────────────────────────────────────────
        self.head = nn.Conv2d(ddcm2_ch, n_classes, kernel_size=1)

        # Initialise non-pretrained weights
        self._init_weights()

    # ── Weight init ───────────────────────────────────────────────────────────

    def _init_weights(self):
        """Kaiming init for all Conv2d layers that are NOT part of the backbone."""
        for module in [self.channel_adapter, self.ddcm1, self.ddcm2, self.head]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, N_channels, H, W)

        Returns:
            logits : (B, n_classes, H, W)  — raw, no sigmoid
        """
        H, W = x.shape[2], x.shape[3]

        # 1. Channel adapter: any-ch → 3ch for ResNet
        x = self.channel_adapter(x)                                 # (B,  3, H,    W)

        # 2. ResNet50 encoder (stride-16)
        x = self.encoder(x)                                         # (B, 1024, H/16, W/16)
        x = self.cbam_enc(x)

        # 3. DDCM block 1  +  ×4 upsample
        x = self.ddcm1(x)                                           # (B, 36, H/16, W/16)
        x = self.cbam_d1(x)
        x = F.interpolate(x, scale_factor=4,
                          mode="bilinear", align_corners=False)      # (B, 36, H/4,  W/4)

        # 4. DDCM block 2  +  ×2 upsample
        x = self.ddcm2(x)                                           # (B, 18, H/4,  W/4)
        x = self.cbam_d2(x)
        x = F.interpolate(x, scale_factor=2,
                          mode="bilinear", align_corners=False)      # (B, 18, H/2,  W/2)

        # 5. Segmentation head  (1×1 conv)
        x = self.head(x)                                            # (B,  1, H/2,  W/2)

        # 6. Final upsample to input resolution
        x = F.interpolate(x, size=(H, W),
                          mode="bilinear", align_corners=False)      # (B,  1, H,    W)

        return x    # ← raw logits; your CombinedLoss already handles sigmoid


# ─────────────────────────────────────────────────────────────────────────────
#  Quick sanity check  (run this file directly)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for n_ch in [1, 3, 4]:
        model = LDDCM_Net(n_channels=n_ch, n_classes=1, use_cbam=False).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

        dummy = torch.randn(2, n_ch, 256, 256, device=device)
        out   = model(dummy)

        print(f"n_channels={n_ch} | params={n_params:.1f}M | "
              f"input={tuple(dummy.shape)} → output={tuple(out.shape)}")

    # CBAM variant
    model_cbam = LDDCM_Net(n_channels=4, n_classes=1, use_cbam=True).to(device)
    n_params   = sum(p.numel() for p in model_cbam.parameters() if p.requires_grad) / 1e6
    dummy      = torch.randn(2, 4, 256, 256, device=device)
    out        = model_cbam(dummy)
    print(f"n_channels=4 +CBAM | params={n_params:.1f}M | "
          f"input={tuple(dummy.shape)} → output={tuple(out.shape)}")