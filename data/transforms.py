import albumentations as A
from albumentations.pytorch import ToTensorV2
from configs.config import CFG

def get_transforms(phase):
    if phase == "train":
        return A.Compose([
            A.RandomCrop(CFG.PATCH_SIZE, CFG.PATCH_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussNoise(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            ToTensorV2(transpose_mask=False),
        ])
    else:
        return A.Compose([
            A.CenterCrop(CFG.PATCH_SIZE, CFG.PATCH_SIZE),
            ToTensorV2(transpose_mask=False),
        ])