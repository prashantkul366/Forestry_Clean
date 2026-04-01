import segmentation_models_pytorch as smp
from configs.config import CFG
import ml_collections


from models.UNext import UNext
from models.UCTransNet import UCTransNet
from models.TransUNet import TransUNet
from models.SwinUnet import SwinUnet
from models.ACC_UNet import ACC_UNet
from models.H_vmunet import H_vmunet
from models.u_kan import UKAN
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

    elif arch == "swinunet":
        m = SwinUnet(
            n_labels=1,
            img_size=CFG.PATCH_SIZE,
            in_chans=CFG.IN_CHANNELS
        )
        
    elif arch == "acc_unet":
        m = ACC_UNet(
            n_channels=CFG.IN_CHANNELS,
            n_classes=1,
            n_filts=32
        )
    elif arch == "h_vmunet":
        m = H_vmunet(
            num_classes=1,
            input_channels=CFG.IN_CHANNELS,   
            c_list=[8,16,32,64,128,256],     
            depths=[2,2,2,2],
            drop_path_rate=0.0,
            bridge=True
        )
    elif arch == "ukan":
        m = UKAN(
            n_classes=1,
            n_channels=CFG.IN_CHANNELS,
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


# import segmentation_models_pytorch as smp
# from configs.config import CFG
# import ml_collections

# # ── Existing custom models ─────────────────────────────────────────────────
# from models.UNext      import UNext
# from models.UCTransNet import UCTransNet
# from models.TransUNet  import TransUNet
# from models.SwinUnet   import SwinUnet
# from models.ACC_UNet   import ACC_UNet
# from models.H_vmunet   import H_vmunet

# # ── New 2023-2025 custom models ────────────────────────────────────────────
# # Copy the 4 new .py files into your  models/  directory, then these work.
# from models.DAEFormer  import DAEFormer
# from models.EGEUNet    import EGEUNet
# from models.VMUNet     import VMUNet
# from models.SAMSeg     import SAMSeg, SAM2Seg


# # =============================================================================
# # build_model
# # =============================================================================

# def build_model():
#     arch = CFG.ARCHITECTURE.lower()

#     # ─────────────────────────────────────────────────────────────────────────
#     # SMP — classic + modern (swap CFG.ENCODER for any backbone below)
#     #
#     # Recommended modern encoders (set in CFG.ENCODER):
#     #   "tu-convnext_base"                  ConvNeXt (CVPR 2022)
#     #   "tu-convnext_large"
#     #   "tu-swin_base_patch4_window7_224"   Swin-B
#     #   "tu-maxvit_base_tf_512"             MaxViT (CVPR 2022)
#     #   "tu-efficientnetv2_m"               EfficientNetV2
#     #   "mit_b4", "mit_b5"                  SegFormer backbones
#     # ─────────────────────────────────────────────────────────────────────────

#     if arch == "unet":
#         m = smp.Unet(
#             encoder_name    = CFG.ENCODER,
#             encoder_weights = "imagenet",
#             in_channels     = CFG.IN_CHANNELS,
#             classes         = 1,
#             activation      = None,
#         )

#     elif arch == "unetplusplus":
#         m = smp.UnetPlusPlus(
#             encoder_name    = CFG.ENCODER,
#             encoder_weights = "imagenet",
#             in_channels     = CFG.IN_CHANNELS,
#             classes         = 1,
#             activation      = None,
#         )

#     elif arch == "segformer":
#         enc = CFG.ENCODER if CFG.ENCODER.startswith("mit") else "mit_b2"
#         m = smp.Segformer(
#             encoder_name    = enc,
#             encoder_weights = "imagenet",
#             in_channels     = CFG.IN_CHANNELS,
#             classes         = 1,
#             activation      = None,
#         )

#     elif arch == "manet":
#         m = smp.MAnet(
#             encoder_name    = CFG.ENCODER,
#             encoder_weights = "imagenet",
#             in_channels     = CFG.IN_CHANNELS,
#             classes         = 1,
#             activation      = None,
#         )

#     elif arch == "fpn":
#         m = smp.FPN(
#             encoder_name    = CFG.ENCODER,
#             encoder_weights = "imagenet",
#             in_channels     = CFG.IN_CHANNELS,
#             classes         = 1,
#             activation      = None,
#         )

#     elif arch == "deeplabv3plus":
#         m = smp.DeepLabV3Plus(
#             encoder_name    = CFG.ENCODER,
#             encoder_weights = "imagenet",
#             in_channels     = CFG.IN_CHANNELS,
#             classes         = 1,
#             activation      = None,
#         )

#     elif arch == "pan":
#         m = smp.PAN(
#             encoder_name    = CFG.ENCODER,
#             encoder_weights = "imagenet",
#             in_channels     = CFG.IN_CHANNELS,
#             classes         = 1,
#             activation      = None,
#         )

#     # ─────────────────────────────────────────────────────────────────────────
#     # Original custom models
#     # ─────────────────────────────────────────────────────────────────────────

#     elif arch == "unext":
#         m = UNext(
#             n_channels = CFG.IN_CHANNELS,
#             n_classes  = 1,
#             img_size   = CFG.PATCH_SIZE,
#         )

#     elif arch == "uctransnet":
#         m = UCTransNet(
#             config     = get_CTranS_config(),
#             n_channels = CFG.IN_CHANNELS,
#             n_classes  = 1,
#             img_size   = CFG.PATCH_SIZE,
#         )

#     elif arch == "transunet":
#         m = TransUNet(
#             n_channels = CFG.IN_CHANNELS,
#             n_classes  = 1,
#             img_size   = CFG.PATCH_SIZE,
#         )

#     elif arch == "swinunet":
#         m = SwinUnet(
#             n_labels = 1,
#             img_size = CFG.PATCH_SIZE,
#             in_chans = CFG.IN_CHANNELS,
#         )

#     elif arch == "acc_unet":
#         m = ACC_UNet(
#             n_channels = CFG.IN_CHANNELS,
#             n_classes  = 1,
#             n_filts    = 32,
#         )

#     elif arch == "h_vmunet":
#         m = H_vmunet(
#             num_classes    = 1,
#             input_channels = CFG.IN_CHANNELS,
#             c_list         = [8, 16, 32, 64, 128, 256],
#             depths         = [2, 2, 2, 2],
#             drop_path_rate = 0.0,
#             bridge         = True,
#         )

#     # ─────────────────────────────────────────────────────────────────────────
#     # NEW: DAEFormer (MICCAI 2023)
#     # Dual Attention Enhanced Transformer
#     # Paper   : https://arxiv.org/abs/2212.13504
#     # Install : pip install timm
#     # ─────────────────────────────────────────────────────────────────────────

#     elif arch == "daeformer":
#         m = DAEFormer(
#             num_classes    = 1,
#             input_channels = CFG.IN_CHANNELS,
#             img_size       = CFG.PATCH_SIZE,
#             # DAEFormer-S (default ~25 M params):
#             embed_dims     = [32, 64, 160, 256],
#             depths         = [2,  2,   2,   2],
#             num_heads      = [1,  2,   5,   8],
#             sr_ratios      = [8,  4,   2,   1],
#             drop_path_rate = 0.1,
#             decoder_dim    = 256,
#         )

#     # ─────────────────────────────────────────────────────────────────────────
#     # NEW: EGE-UNet (2023)
#     # Element-wise Gradient Enhancement UNet  (~50 K params)
#     # Paper   : https://arxiv.org/abs/2307.08473
#     # Install : none (pure PyTorch)
#     # ─────────────────────────────────────────────────────────────────────────

#     elif arch == "egeunet":
#         m = EGEUNet(
#             num_classes    = 1,
#             input_channels = CFG.IN_CHANNELS,
#             c_list         = [8, 16, 24, 32, 48, 64],  # ~50 K params
#             bridge         = True,
#             # gt_ds=True  → auxiliary outputs are weighted-summed into
#             # a single tensor before returning, so the training loop
#             # is unchanged.  Set False to disable deep supervision.
#             gt_ds          = True,
#         )

#     # ─────────────────────────────────────────────────────────────────────────
#     # NEW: VM-UNet (2024)
#     # Vision Mamba UNet — linear-complexity SSM encoder/decoder
#     # Paper   : https://arxiv.org/abs/2402.02491
#     # Install : pip install mamba-ssm causal-conv1d  (requires CUDA)
#     #           Falls back to a GRU automatically if mamba-ssm is missing.
#     # ─────────────────────────────────────────────────────────────────────────

#     elif arch == "vmunet":
#         m = VMUNet(
#             input_channels = CFG.IN_CHANNELS,
#             num_classes    = 1,
#             # Lighter variant — change to [96,192,384,768] for full paper size:
#             dims           = [48,  96, 192, 384],
#             depths         = [2,   2,   9,   2],
#             depths_decoder = [2,   9,   2,   2],
#             drop_path_rate = 0.1,
#         )

#     # ─────────────────────────────────────────────────────────────────────────
#     # NEW: SAMSeg (2023)
#     # SAM ViT-B image encoder (frozen) + lightweight FPN decoder
#     # Paper   : https://arxiv.org/abs/2304.02643
#     # Install : pip install git+https://github.com/facebookresearch/segment-anything.git
#     # Ckpt    : wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
#     #
#     # Add to your config.py:
#     #   CFG.SAM_CHECKPOINT   = "checkpoints/sam_vit_b_01ec64.pth"
#     #   CFG.SAM_MODEL_TYPE   = "vit_b"        # vit_b | vit_l | vit_h
#     #   CFG.SAM_FREEZE_ENCODER = True         # only train the decoder
#     #   CFG.SAM_LORA_RANK    = 4             # 0 = no LoRA; >0 = efficient fine-tune
#     # ─────────────────────────────────────────────────────────────────────────

#     elif arch == "samseg":
#         m = SAMSeg(
#             num_classes    = 1,
#             input_channels = CFG.IN_CHANNELS,
#             model_type     = getattr(CFG, "SAM_MODEL_TYPE",     "vit_b"),
#             checkpoint     = getattr(CFG, "SAM_CHECKPOINT",     None),
#             freeze_encoder = getattr(CFG, "SAM_FREEZE_ENCODER", True),
#             lora_rank      = getattr(CFG, "SAM_LORA_RANK",      0),
#             decoder_hidden = 128,
#         )

#     # ─────────────────────────────────────────────────────────────────────────
#     # NEW: SAM2Seg (2024)
#     # SAM2 Hiera encoder (frozen) + FPN decoder
#     # Paper   : https://arxiv.org/abs/2408.00714
#     # Install : pip install git+https://github.com/facebookresearch/sam2.git
#     #
#     # Add to your config.py:
#     #   CFG.SAM2_MODEL_CFG  = "sam2_hiera_tiny"   # tiny|small|base+|large
#     #   CFG.SAM2_CHECKPOINT = "checkpoints/sam2_hiera_tiny.pt"
#     # ─────────────────────────────────────────────────────────────────────────

#     elif arch == "sam2seg":
#         m = SAM2Seg(
#             num_classes    = 1,
#             input_channels = CFG.IN_CHANNELS,
#             model_cfg      = getattr(CFG, "SAM2_MODEL_CFG",     "sam2_hiera_tiny"),
#             checkpoint     = getattr(CFG, "SAM2_CHECKPOINT",    None),
#             freeze_encoder = getattr(CFG, "SAM_FREEZE_ENCODER", True),
#             decoder_hidden = 128,
#         )

#     else:
#         available = (
#             "unet, unetplusplus, segformer, manet, fpn, deeplabv3plus, pan, "
#             "unext, uctransnet, transunet, swinunet, acc_unet, h_vmunet, "
#             "daeformer, egeunet, vmunet, samseg, sam2seg"
#         )
#         raise ValueError(f"Unknown arch: '{arch}'\nAvailable: {available}")

#     return m.to(CFG.DEVICE)


# # =============================================================================
# # CTrans config (for UCTransNet)
# # =============================================================================

# def get_CTranS_config():
#     config = ml_collections.ConfigDict()
#     config.transformer                         = ml_collections.ConfigDict()
#     config.KV_size                             = 960
#     config.transformer.num_heads               = 4
#     config.transformer.num_layers              = 4
#     config.expand_ratio                        = 4
#     config.transformer.embeddings_dropout_rate = 0.1
#     config.transformer.attention_dropout_rate  = 0.1
#     config.transformer.dropout_rate            = 0
#     config.patch_sizes                         = [16, 8, 4, 2]
#     config.base_channel                        = 64
#     config.n_classes                           = 1
#     return config