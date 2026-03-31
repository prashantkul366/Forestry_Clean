# nets/TransUNet.py
import torch
import torch.nn as nn
import numpy as np

# use your provided TransUNet implementation bits
from .vit_seg_modelling import VisionTransformer, CONFIGS

# pull img_size from your global Config so we don't change train_model.py
try:
    import Config as GLOBAL_CONFIG
    _GLOBAL_IMG_SIZE = getattr(GLOBAL_CONFIG, "img_size", 224)
except Exception:
    _GLOBAL_IMG_SIZE = 224


class TransUNet(nn.Module):
    """
    Standalone TransUNet wrapper.
    - Signature matches how other models are constructed in train_model.py:
        TransUNet(n_channels=..., n_classes=...)
    - For binary tasks (n_classes==1) returns sigmoid probabilities to match WeightedDiceBCE.
    - Auto-loads ViT weights from vit_seg_configs.py if available.
    """
    def __init__(self, n_channels=3, n_classes=1, backbone='R50-ViT-B_16', img_size=None, vis=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self._binary = (n_classes == 1)

        # pick a backbone: 'ViT-B_16' (pure ViT) or 'R50-ViT-B_16' (ResNet50+ViT)
        cfg = CONFIGS[backbone]
        print("Transunet Initiated")
        # decide img_size (must be divisible by patch size, typically 16)
        if img_size is None:
            img_size = _GLOBAL_IMG_SIZE
        patch = cfg.patches.size[0] if hasattr(cfg.patches, "size") else 16
        assert img_size % patch == 0, f"img_size {img_size} must be divisible by patch size {patch}"

        # set class count (use 1 for binary)
        cfg.n_classes = n_classes

        # build the TransUNet (VisionTransformer acts as the TransUNet in this repo)
        self.vit = VisionTransformer(cfg, img_size=img_size, zero_head=False, vis=vis)

        # try to load pretrained weights if the path is set in vit_seg_configs
        self._load_pretrained_if_available()

    def _load_pretrained_if_available(self):
        path = getattr(self.vit.config, 'pretrained_path', None)
        if path:
            try:
                weights = np.load(path)
                self.vit.load_from(weights)
            except Exception as e:
                print(f"[TransUNet] pretrained load skipped: {e}")

    def forward(self, x):
        logits = self.vit(x)  # (B, C, H, W)
        # if self._binary:
        #     return torch.sigmoid(logits)  # matches your WeightedDiceBCE (expects probs)
        # return logits
        return logits
