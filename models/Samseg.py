# =============================================================================
# SAMSeg — SAM Image Encoder + Lightweight U-Net Decoder
# for supervised binary / multi-class segmentation
#
# Strategy
#   • Load SAM's ViT image encoder (frozen by default)
#   • Add a trainable FPN-style decoder on top
#   • Train ONLY the decoder (+ optional LoRA adapters) on your data
#   • Works with SAM ViT-B (~86 M params encoder) or ViT-L / ViT-H
#
# Install:
#   pip install git+https://github.com/facebookresearch/segment-anything.git
#
# Checkpoints (download once):
#   ViT-B  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth  (~375 MB)
#   ViT-L  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth  (~1.2 GB)
#   ViT-H  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth  (~2.4 GB)
#
# SAM2 variant (SAM2Seg):
#   pip install git+https://github.com/facebookresearch/sam2.git
#   or: pip install ultralytics  (has SAM2 built-in)
#
# Notes
#   • SAM was trained on 1024×1024 RGB images.
#     For other sizes/channels the encoder input is auto-resized to 1024×1024.
#   • If input has != 3 channels, a learnable adapter conv is inserted
#     BEFORE the SAM encoder to project to 3 channels.
#   • LoRA adapters can be inserted into the encoder for efficient fine-tuning.
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── SAM1 import ───────────────────────────────────────────────────────────────
try:
    from segment_anything import sam_model_registry
    _SAM1_OK = True
except ImportError:
    _SAM1_OK = False

# ── SAM2 import ───────────────────────────────────────────────────────────────
try:
    from sam2.build_sam import build_sam2
    _SAM2_OK = True
except ImportError:
    _SAM2_OK = False


# =============================================================================
# Shared decoder (used by both SAMSeg and SAM2Seg)
# =============================================================================

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class DecodeBlock(nn.Module):
    """Upsample-2× + skip-cat + 2×Conv."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch + skip_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class LightDecoder(nn.Module):
    """
    4-stage FPN decoder.
    Takes a list of feature maps (coarsest → finest) and
    progressively upsamples to the target resolution.
    """
    def __init__(self, feat_chs: list, hidden: int = 128, num_classes: int = 1):
        """
        feat_chs : list of input channel counts, coarsest first
                   e.g. [256, 256, 256, 256] for SAM1 single-scale output
                       or [256, 64, 32] for SAM2 FPN
        """
        super().__init__()
        self.stages = nn.ModuleList()
        in_ch = feat_chs[0]
        for skip_ch in feat_chs[1:]:
            self.stages.append(DecodeBlock(in_ch, skip_ch, hidden))
            in_ch = hidden
        # final stage: no skip
        self.stages.append(DecodeBlock(in_ch, 0, hidden // 2))
        self.head = nn.Conv2d(hidden // 2, num_classes, 1)

    def forward(self, features: list, out_hw: tuple) -> torch.Tensor:
        x = features[0]
        for i, stage in enumerate(self.stages):
            skip = features[i + 1] if i + 1 < len(features) else None
            x    = stage(x, skip)
        return F.interpolate(self.head(x), size=out_hw,
                             mode="bilinear", align_corners=False)


# =============================================================================
# LoRA Adapter (optional efficient fine-tuning of SAM encoder)
# =============================================================================

class LoRALinear(nn.Module):
    """
    LoRA wrapper for nn.Linear.
    Inserts low-rank matrices A and B (rank r).
    Only A and B are trained; original weight is frozen.
    """
    def __init__(self, linear: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        d_in, d_out = linear.in_features, linear.out_features
        self.linear = linear
        self.scale  = alpha / r
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r))

    def forward(self, x):
        return self.linear(x) + (x @ self.A.T @ self.B.T) * self.scale


def add_lora_to_encoder(encoder: nn.Module, r: int = 4):
    """Replace all query/value projections in ViT attention with LoRA versions."""
    for module in encoder.modules():
        if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
            module.qkv = LoRALinear(module.qkv, r=r)
        if hasattr(module, "proj") and isinstance(module.proj, nn.Linear):
            module.proj = LoRALinear(module.proj, r=r)


# =============================================================================
# SAMSeg — SAM1 (ViT) encoder + FPN decoder
# =============================================================================

class SAMSeg(nn.Module):
    """
    SAMSeg: SAM image encoder + trainable FPN decoder.

    Args
    ----
    num_classes    : output channels (1 for binary seg)
    input_channels : input image channels
                     if != 3, a 1×1 learnable conv adapts to 3 before SAM encoder
    model_type     : "vit_b" | "vit_l" | "vit_h"
    checkpoint     : path to SAM .pth checkpoint file
                     e.g. "checkpoints/sam_vit_b_01ec64.pth"
    freeze_encoder : freeze SAM encoder weights (True = only train decoder)
    lora_rank      : if > 0, add LoRA adapters to encoder attention (fine-tuning)
    decoder_hidden : channel width of the FPN decoder
    sam_input_size : SAM encoder expects 1024 by default; change only if you
                     know what you are doing
    """
    def __init__(
        self,
        num_classes    : int   = 1,
        input_channels : int   = 3,
        model_type     : str   = "vit_b",
        checkpoint     : str   = None,
        freeze_encoder : bool  = True,
        lora_rank      : int   = 0,
        decoder_hidden : int   = 128,
        sam_input_size : int   = 1024,
    ):
        super().__init__()

        if not _SAM1_OK:
            raise ImportError(
                "segment-anything is not installed.\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        self.sam_input_size  = sam_input_size
        self.input_channels  = input_channels

        # ── Channel adapter (if input != 3 ch) ──────────────────────────────
        if input_channels != 3:
            self.chan_adapt = nn.Conv2d(input_channels, 3, 1, bias=False)
        else:
            self.chan_adapt = None

        # ── SAM encoder ──────────────────────────────────────────────────────
        if checkpoint is None:
            import warnings
            warnings.warn(
                "SAMSeg: no checkpoint provided — encoder will have random weights.\n"
                "  Download: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                stacklevel=2,
            )
            sam = sam_model_registry[model_type](checkpoint=None)
        else:
            sam = sam_model_registry[model_type](checkpoint=checkpoint)

        self.encoder = sam.image_encoder  # ViT — outputs [B, 256, 64, 64] for 1024 input

        # ── LoRA adapters (optional) ─────────────────────────────────────────
        if lora_rank > 0:
            add_lora_to_encoder(self.encoder, r=lora_rank)

        # ── Freeze encoder ───────────────────────────────────────────────────
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            # unfreeze only LoRA params if any
            if lora_rank > 0:
                for m in self.encoder.modules():
                    if isinstance(m, LoRALinear):
                        m.A.requires_grad_(True)
                        m.B.requires_grad_(True)

        # ── Decoder ──────────────────────────────────────────────────────────
        # SAM encoder outputs a single 256-ch feature map; we build a
        # 3-stage decoder from 64×64 → 128×128 → 256×256 → output_size
        enc_out_ch = 256  # ViT-B / L / H all output 256 channels
        self.decoder = LightDecoder(
            feat_chs  = [enc_out_ch],   # single-scale from SAM
            hidden    = decoder_hidden,
            num_classes = num_classes,
        )

        # ── Final head stages ─────────────────────────────────────────────────
        # We need more upsample stages manually since SAM→64×64 is very coarse
        upsample_stages = []
        ch = decoder_hidden // 2
        for _ in range(3):          # 64→128→256→512
            upsample_stages.append(ConvBNReLU(ch, ch // 2))
            ch = ch // 2
        self.upsample_extra = nn.ModuleList(upsample_stages)
        self.final_head = nn.Conv2d(ch, num_classes, 1)

        self._init_decoder_weights()

    def _init_decoder_weights(self):
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _resize_for_sam(self, x: torch.Tensor) -> torch.Tensor:
        """Resize to SAM's expected input size."""
        return F.interpolate(x.float(), size=(self.sam_input_size, self.sam_input_size),
                             mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Channel adapt
        if self.chan_adapt is not None:
            x = self.chan_adapt(x)

        # 2. Resize to SAM's input size (1024×1024)
        x_sam = self._resize_for_sam(x)

        # 3. SAM encoder → [B, 256, 64, 64]
        feats = self.encoder(x_sam)              # [B, 256, 64, 64]

        # 4. Decode (LightDecoder does one initial upsample stage)
        x = self.decoder([feats], out_hw=(64, 64))   # stays at 64 initially

        # 5. Progressive upsample to original size
        for stage in self.upsample_extra:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = stage(x)

        x = self.final_head(x)
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)


# =============================================================================
# SAM2Seg — SAM2 Hiera encoder + FPN decoder
# =============================================================================

class SAM2Seg(nn.Module):
    """
    SAM2Seg: SAM2 image encoder (Hiera) + trainable FPN decoder.

    SAM2's Hiera backbone outputs multi-scale features, making it a
    natural fit for a dense prediction decoder.

    Args
    ----
    num_classes    : output channels
    input_channels : if != 3, a learnable adapter is added
    model_cfg      : SAM2 model config string
                     "sam2_hiera_tiny"  | "sam2_hiera_small"
                     "sam2_hiera_base+" | "sam2_hiera_large"
    checkpoint     : path to SAM2 .pt checkpoint
    freeze_encoder : freeze SAM2 encoder
    decoder_hidden : FPN decoder channel width
    """
    def __init__(
        self,
        num_classes    : int  = 1,
        input_channels : int  = 3,
        model_cfg      : str  = "sam2_hiera_tiny",
        checkpoint     : str  = None,
        freeze_encoder : bool = True,
        decoder_hidden : int  = 128,
    ):
        super().__init__()

        if not _SAM2_OK:
            raise ImportError(
                "sam2 is not installed.\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git\n"
                "  or:  pip install ultralytics  (includes SAM2)"
            )

        if input_channels != 3:
            self.chan_adapt = nn.Conv2d(input_channels, 3, 1, bias=False)
        else:
            self.chan_adapt = None

        # ── SAM2 model ────────────────────────────────────────────────────────
        cfg_path = f"configs/sam2/{model_cfg}.yaml"
        sam2     = build_sam2(cfg_path, checkpoint, device="cpu")
        self.backbone = sam2.image_encoder          # Hiera encoder
        self.fpn      = sam2.image_encoder.neck     # FPN neck

        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # SAM2 FPN outputs: [B, 256, H/4, W/4], [B, 64, H/8, W/8], [B, 32, H/16, W/16]
        # (sizes depend on the model variant)
        self.decoder = LightDecoder(
            feat_chs    = [256, 64, 32],
            hidden      = decoder_hidden,
            num_classes = num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.chan_adapt is not None:
            x = self.chan_adapt(x)

        # SAM2 image encoder returns backbone_fpn feature list
        out         = self.backbone(x)
        fpn_feats   = out["backbone_fpn"]           # list of feature maps
        # coarsest → finest for the decoder
        feats       = list(reversed(fpn_feats))     # [256-ch, 64-ch, 32-ch, ...]
        return self.decoder(feats, (H, W))