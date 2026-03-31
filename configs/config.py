import os
import torch

class CFG:
    BASE = "/content/drive/MyDrive/Prashant/Forestry_data/data_new/dataset_small_bal"

    TRAIN_IMGS  = f"{BASE}/train/images"
    TRAIN_MASKS = f"{BASE}/train/masks"
    VAL_IMGS    = f"{BASE}/val/images"
    VAL_MASKS   = f"{BASE}/val/masks"

    PATCH_SIZE  = 256
    BATCH_SIZE  = 8
    NUM_WORKERS = 2
    EPOCHS      = 200
    LR          = 3e-4
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

    PATIENCE = 50

    ENCODER      = "mit_b2"
    IN_CHANNELS  = 4
    ARCHITECTURE = "segformer"

    DICE_WEIGHT = 0.5
    BCE_WEIGHT  = 0.5
    FOCAL_GAMMA = 2.0
    POS_WEIGHT  = 6.0

    ROAD_RATIO      = 0.70
    ROAD_MIN_PIXELS = 30

    SAVE_DIR = "/content/drive/MyDrive/Prashant/Forestry_data/CLEAN"
    os.makedirs(SAVE_DIR, exist_ok=True)