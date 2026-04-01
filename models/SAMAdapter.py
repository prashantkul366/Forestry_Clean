# models/SAMAdapter.py
"""
SAM-Adapter: Adapting Segment Anything in Underperformed Scenes
Paper : ICCVW 2023 — Chen et al.
GitHub: https://github.com/tianrun-chen/SAM-Adapter-PyTorch

Requirements
────────────
pip install git+https://github.com/facebookresearch/segment-anything.git

Checkpoints (pick one — ViT-B is recommended for memory):
  ViT-B (~375 MB):
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
  ViT-L (~1.2 GB):
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

Add to configs/config.py:
    SAM_CHECKPOINT  = "pretrained_ckpt/sam_vit_b_01ec64.pth"
    SAM_MODEL_TYPE  = "vit_b"   # "vit_b" | "vit_l" | "vit_h"

Architecture (Fig. 1 in paper)
───────────────────────────────
4-ch input
  → ChannelProjection          → 3-ch image for SAM
  → HighFreqExtractor          → Fhfc  (high-frequency patch features)
  │
  SAM ImageEncoder (FROZEN)
    PatchEmbed                 → Fpe   (patch-embedding features)
    Fi = Fpe + Fhfc            (Eq. 3 in paper)
    for each transformer block i:
        x = block(x)
        Pi = MLP_up( GELU( MLP_tune_i( Fi ) ) )   (Eq. 1)
        x = x + Pi             ← adapter injection
    neck                       → image_embeddings (B, 256, H/64, W/64)
  │
  SAM MaskDecoder (TUNABLE)
    no-prompt dense embedding
    → low_res_masks  (B, 1, H/4, W/4)
  │
  Bilinear upsample            → logits  (B, 1, H, W)   ← raw, no sigmoid

What is frozen vs tunable
─────────────────────────
  Frozen  : SAM image encoder (ViT backbone)
  Tunable : channel projection, high-freq extractor, all adapters,
            shared MLP_up, SAM mask decoder, no_mask_embed
"""

import os
import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from segment_anything import sam_model_registry
    from segment_anything.modeling import Sam
    _SAM_AVAILABLE = True
except ImportError:
    _SAM_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Channel adapter  (4ch LiDAR → 3ch for SAM)
# ─────────────────────────────────────────────────────────────────────────────

class ChannelProjection(nn.Module):
    """
    Project arbitrary input channels to 3 for SAM.
    Uses 3×3 conv for spatial mixing (better than 1×1 for multi-channel data).
    """
    def __init__(self, in_channels: int):
        super().__init__()
        if in_channels == 3:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 3, 1, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. High-frequency extractor  (Fhfc)
# ─────────────────────────────────────────────────────────────────────────────

class HighFreqExtractor(nn.Module):
    """
    Fhfc from §4.2: high-frequency components of the input image.
    Implementation: high-freq = input − Gaussian-smoothed input,
    then project to embed_dim at patch resolution.

    Why it helps for roads/vessels:
      Edges and thin structures ARE the high-frequency signal.
      This gives the adapter explicit structural cues that SAM's
      plain ViT encoder may miss.
    """
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 16):
        super().__init__()
        # Approximate Gaussian smoothing with AvgPool
        self.smooth = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        # Project to patch-level feature map
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3,
                      padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim,
                      kernel_size=patch_size, stride=patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hfc = x - self.smooth(x)           # (B, C, H, W)
        return self.proj(hfc)              # (B, embed_dim, H/ps, W/ps)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Adapter layer  (Eq. 1 in paper)
# ─────────────────────────────────────────────────────────────────────────────

class AdapterLayer(nn.Module):
    """
    Layer-specific MLP_tune_i.
    Shared MLP_up is passed in at forward time (lives in SAMAdapterEncoder).

    Pi = MLP_up( GELU( MLP_tune_i( Fi ) ) )
    """
    def __init__(self, task_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.mlp_tune = nn.Linear(task_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, fi: torch.Tensor, mlp_up: nn.Linear) -> torch.Tensor:
        # fi : (B, N, task_dim)   N = H/ps * W/ps tokens
        return mlp_up(self.act(self.mlp_tune(fi)))  # (B, N, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Adapted encoder — wraps SAM's image encoder with adapter injection
# ─────────────────────────────────────────────────────────────────────────────

class SAMAdaptedEncoder(nn.Module):
    """
    Wraps SAM's ImageEncoderViT and injects one adapter per transformer block.

    Frozen: all SAM encoder parameters.
    Tunable: adapter layers + shared MLP_up.
    """

    def __init__(
        self,
        sam_encoder,
        num_blocks: int,
        embed_dim: int,
        task_dim: int,
        adapter_hidden: int = 32,
    ):
        super().__init__()
        self.encoder = sam_encoder

        # One adapter per transformer block
        self.adapters = nn.ModuleList([
            AdapterLayer(task_dim, adapter_hidden)
            for _ in range(num_blocks)
        ])

        # Shared up-projection (maps hidden_dim → embed_dim)
        self.mlp_up = nn.Linear(adapter_hidden, embed_dim)

        # Freeze SAM encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        x: torch.Tensor,                # (B, 3, H, W)
        task_features: torch.Tensor,    # (B, task_dim, Hp, Wp) patch-level
    ) -> torch.Tensor:
        """Returns image embeddings (B, out_chans, H/64, W/64)."""
        enc = self.encoder

        # ── Patch embedding ──────────────────────────────────────────────────
        x = enc.patch_embed(x)           # (B, Hp, Wp, embed_dim)

        if enc.pos_embed is not None:
            x = x + enc.pos_embed        # add absolute positional embedding

        B, Hp, Wp, C = x.shape

        # ── Resize task features to match patch grid ──────────────────────────
        tf = F.interpolate(
            task_features, size=(Hp, Wp),
            mode='bilinear', align_corners=False,
        )                                # (B, task_dim, Hp, Wp)
        tf_flat = tf.flatten(2).transpose(1, 2)  # (B, Hp*Wp, task_dim)

        # ── Transformer blocks + adapter injection ────────────────────────────
        for i, block in enumerate(enc.blocks):
            x = block(x)                 # (B, Hp, Wp, C) — SAM block format

            # Inject adapter: Pi added to transformer output
            x_flat = x.reshape(B, Hp * Wp, C)
            pi = self.adapters[i](tf_flat, self.mlp_up)   # (B, Hp*Wp, C)
            x = (x_flat + pi).reshape(B, Hp, Wp, C)

        # ── Neck (norm + 1×1/3×3 convs) → (B, out_chans, Hp, Wp) ───────────
        x = enc.neck(x.permute(0, 3, 1, 2))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full SAM-Adapter model
# ─────────────────────────────────────────────────────────────────────────────

class SAMAdapterSeg(nn.Module):
    """
    SAM-Adapter for binary segmentation.

    Args:
        n_channels    : input channels (4 for your LiDAR data)
        n_classes     : output channels (1 for binary seg)
        checkpoint    : path to SAM .pth file (None → random init for testing)
        model_type    : "vit_b" | "vit_l" | "vit_h"
        adapter_hidden: hidden dim of MLP_tune (paper uses 32)
        img_size      : SAM input size (paper uses 1024;  256 is faster
                        but requires pos_embed interpolation — see notes)

    Forward returns raw logits (B, n_classes, H, W) — no sigmoid.
    """

    # SAM config per model type
    _SAM_CONFIGS = {
        "vit_b": {"embed_dim": 768,  "depth": 12, "patch_size": 16},
        "vit_l": {"embed_dim": 1024, "depth": 24, "patch_size": 16},
        "vit_h": {"embed_dim": 1280, "depth": 32, "patch_size": 16},
    }

    def __init__(
        self,
        n_channels:     int   = 4,
        n_classes:      int   = 1,
        checkpoint:     Optional[str] = None,
        model_type:     str   = "vit_b",
        adapter_hidden: int   = 32,
        img_size:       int   = 1024,
    ):
        super().__init__()

        if not _SAM_AVAILABLE:
            raise ImportError(
                "segment_anything is required.\n"
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        cfg = self._SAM_CONFIGS[model_type]
        embed_dim  = cfg["embed_dim"]
        patch_size = cfg["patch_size"]
        num_blocks = cfg["depth"]

        self.img_size  = img_size
        self.n_classes = n_classes
        self.prompt_embed_dim = 256   # SAM constant

        # ── 1. Channel projection ─────────────────────────────────────────────
        self.channel_proj = ChannelProjection(n_channels)

        # ── 2. High-frequency extractor (Fhfc) ────────────────────────────────
        self.hfc_extractor = HighFreqExtractor(
            in_channels=n_channels,   # computed on original multi-channel input
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        # ── 3. Load SAM ───────────────────────────────────────────────────────
        sam: Sam = self._load_sam(model_type, checkpoint, img_size)

        # ── 4. Adapted encoder ────────────────────────────────────────────────
        # task_dim = embed_dim (Fpe + Fhfc both projected to embed_dim)
        self.adapted_encoder = SAMAdaptedEncoder(
            sam_encoder   = sam.image_encoder,
            num_blocks    = num_blocks,
            embed_dim     = embed_dim,
            task_dim      = embed_dim,
            adapter_hidden= adapter_hidden,
        )

        # ── 5. SAM mask decoder (tunable) ─────────────────────────────────────
        self.mask_decoder = sam.mask_decoder
        for p in self.mask_decoder.parameters():
            p.requires_grad_(True)

        # ── 6. Dense positional encoding ──────────────────────────────────────
        # image_embedding_size = img_size // patch_size (e.g. 1024//16 = 64)
        self.image_embedding_size = img_size // patch_size
        self.pe_layer = sam.prompt_encoder.pe_layer

        # No-prompt dense embedding (trainable)
        self.no_mask_embed = nn.Embedding(1, self.prompt_embed_dim)

        # ── 7. Patch-embedding projection for Fpe ─────────────────────────────
        # SAM's patch_embed output has embed_dim channels already — we just
        # need a 1×1 conv to store a "copy" before the blocks run.
        # We extract Fpe inside forward() from the patch_embed directly.

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_sam(model_type: str, checkpoint: Optional[str], img_size: int) -> "Sam":
        """Load SAM, interpolating pos_embed if img_size != 1024."""
        if checkpoint is None or not os.path.exists(checkpoint):
            print(f"⚠️  No SAM checkpoint found at '{checkpoint}'. "
                  "Loading random init (testing only).")
            sam = sam_model_registry[model_type](checkpoint=None)
        else:
            print(f"✅ Loading SAM {model_type} from {checkpoint}")
            sam = sam_model_registry[model_type](checkpoint=checkpoint)

        # Interpolate pos_embed if img_size differs from pretrained 1024
        if img_size != 1024 and sam.image_encoder.pos_embed is not None:
            pos = sam.image_encoder.pos_embed  # (1, H, W, C)
            H_new = img_size // 16
            pos_interp = F.interpolate(
                pos.permute(0, 3, 1, 2).float(),
                size=(H_new, H_new),
                mode='bicubic', align_corners=False,
            ).permute(0, 2, 3, 1)
            sam.image_encoder.pos_embed = nn.Parameter(pos_interp)
            print(f"   pos_embed interpolated to {H_new}×{H_new}")

        return sam

    def _get_dense_pe(self) -> torch.Tensor:
        """Positional encoding for the image embedding grid."""
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, n_channels, H, W)   — your 4-channel LiDAR patches

        Returns:
            logits : (B, n_classes, H, W)  — raw, no sigmoid
        """
        B, _, H, W = x.shape

        # ── Resize to SAM's expected input size ───────────────────────────────
        # SAM's encoder was pretrained at img_size × img_size.
        # We resize here and upsample the output back at the end.
        if H != self.img_size or W != self.img_size:
            x_sam_input = F.interpolate(
                x, size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False,
            )
        else:
            x_sam_input = x

        # ── Extract task features Fi = Fpe + Fhfc ────────────────────────────
        # Fhfc: computed on the input (original channels, resized)
        x_hfc_input = F.interpolate(
            x, size=(self.img_size, self.img_size),
            mode='bilinear', align_corners=False,
        )
        fhfc = self.hfc_extractor(x_hfc_input)   # (B, embed_dim, Hp, Wp)

        # Fpe: output of SAM's patch_embed (before blocks)
        x3 = self.channel_proj(x_sam_input)       # (B, 3, img_size, img_size)
        fpe = self.adapted_encoder.encoder.patch_embed(x3)  # (B, Hp, Wp, embed_dim)
        fpe = fpe.permute(0, 3, 1, 2)             # (B, embed_dim, Hp, Wp)

        # Combine (Eq. 3)
        task_features = fpe + fhfc                # (B, embed_dim, Hp, Wp)

        # ── Adapted SAM encoder ───────────────────────────────────────────────
        image_embeddings = self.adapted_encoder(x3, task_features)
        # (B, 256, Hp, Wp)  where Hp = img_size // 16

        # ── SAM mask decoder (no user prompts) ────────────────────────────────
        sparse_embeddings = torch.empty(
            (B, 0, self.prompt_embed_dim), device=x.device
        )
        dense_embeddings = (
            self.no_mask_embed.weight
            .reshape(1, -1, 1, 1)
            .expand(B, -1,
                    self.image_embedding_size,
                    self.image_embedding_size)
        )

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self._get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        # low_res_masks: (B, 1, img_size/4, img_size/4)

        # ── Upsample to original input resolution ────────────────────────────
        logits = F.interpolate(
            low_res_masks, size=(H, W),
            mode='bilinear', align_corners=False,
        )   # (B, 1, H, W)

        return logits   # raw logits — CombinedLoss handles sigmoid internally