import segmentation_models_pytorch as smp
from configs.config import CFG
def build_model():
    arch = CFG.ARCHITECTURE.lower()
    if arch == "unet":
        m = smp.Unet(encoder_name=CFG.ENCODER, encoder_weights="imagenet",
                     in_channels=CFG.IN_CHANNELS, classes=1, activation=None)
    elif arch == "unetplusplus":
        m = smp.UnetPlusPlus(encoder_name=CFG.ENCODER, encoder_weights="imagenet",
                             in_channels=CFG.IN_CHANNELS, classes=1, activation=None)
    elif arch == "segformer":
        m = smp.Segformer(encoder_name="mit_b2", encoder_weights="imagenet",
                          in_channels=CFG.IN_CHANNELS, classes=1, activation=None)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return m.to(CFG.DEVICE)
