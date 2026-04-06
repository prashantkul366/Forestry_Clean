import os
# from sympy import content
import torch

class CFG:
    # BASE = "/content/drive/MyDrive/Prashant/Forestry_data/data_new/dataset_small_bal"
    BASE = "/content/drive/MyDrive/Prashant/Forestry_data/data_new/dataset_clean"

    SEED = 42
    TRAIN_IMGS  = f"{BASE}/train/images"
    TRAIN_MASKS = f"{BASE}/train/masks"
    VAL_IMGS    = f"{BASE}/val/images"
    VAL_MASKS   = f"{BASE}/val/masks"


    PATCH_SIZE = 256
    IMG_SIZE   = 256
    # IMG_SIZE   = 224 
    BATCH_SIZE  = 8
    # BATCH_SIZE  = 4
    # BATCH_SIZE  = 2
    NUM_WORKERS = 2
    EPOCHS      = 100
    # EPOCHS      = 10
    LR          = 3e-4
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

    PATIENCE = 50
    # PATIENCE = 5

    # ENCODER      = "mit_b2"
    ENCODER      = "resnet34"        # or "mit_b2" for SegFormer



    # SAM_CHECKPOINT  = "content/drive/MyDrive/Prashant/Pretrain/sam_vit_b_01ec64.pth"
    SAM_CHECKPOINT  = "/content/drive/MyDrive/Prashant/Pretrain/sam_vit_b_01ec64.pth"
    SAM_MODEL_TYPE  = "vit_b"
    IN_CHANNELS  = 4



    ARCHITECTURE = "unet"
    # ARCHITECTURE = "unetplusplus" 
    # ARCHITECTURE = "segformer"
    # ARCHITECTURE = "unext"
    # ARCHITECTURE = "uctransnet"
    # ARCHITECTURE = "transunet"
    # ARCHITECTURE = "swinunet"
    # ARCHITECTURE = "acc_unet"
    # ARCHITECTURE = "h_vmunet"
    # ARCHITECTURE = "egeunet"
    # ARCHITECTURE = "samseg" 
    # ARCHITECTURE = "ukan"
    # ARCHITECTURE = "lddcm"
    # ARCHITECTURE = "dscnet"
    # ARCHITECTURE = "fr_unet"
    # ARCHITECTURE    = "sam_adapter"

    RESUME = False
    RESUME_PATH = None  # or specific checkpoint path
    
    DICE_WEIGHT = 0.5
    BCE_WEIGHT  = 0.5
    FOCAL_GAMMA = 2.0
    # POS_WEIGHT  = 6.0
    POS_WEIGHT  = 11.7

    ROAD_RATIO      = 0.70
    ROAD_MIN_PIXELS = 30

    SAVE_DIR = "/content/drive/MyDrive/Prashant/Forestry_data/CLEAN_ORIGINAL"
    os.makedirs(SAVE_DIR, exist_ok=True)



