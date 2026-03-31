import segmentation_models_pytorch as smp
from configs.config import CFG
from models.UNext import UNext

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
    elif arch == "unext":
        m = UNext(n_channels=CFG.IN_CHANNELS, n_classes=1,img_size=CFG.PATCH_SIZE)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return m.to(CFG.DEVICE)
