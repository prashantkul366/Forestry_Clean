import segmentation_models_pytorch as smp
from configs.config import CFG
import ml_collections


from models.UNext import UNext
from models.UCTransNet import UCTransNet
from nets.TransUNet import TransUNet
# from configs.config import UCTransNetConfig

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
    
    elif arch == "uctransnet":
        config = get_CTranS_config()   
        m = UCTransNet(
            config=config, n_channels=CFG.IN_CHANNELS, 
            n_classes=1, img_size=CFG.PATCH_SIZE
        )

    elif arch == "transunet":
    m = TransUNet(
        n_channels=CFG.IN_CHANNELS,  
        n_classes=1,
        img_size=CFG.PATCH_SIZE
    )
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return m.to(CFG.DEVICE)



##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config