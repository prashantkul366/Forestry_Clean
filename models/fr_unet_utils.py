# models/fr_unet_utils.py
import torch.nn as nn
from timm.models.layers import trunc_normal_


class InitWeights_He:
    """
    Kaiming He initialisation for conv layers, trunc_normal for Linear,
    and constant init for LayerNorm — matches the original FR-UNet repo.
    """
    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d,
                               nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.neg_slope)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)