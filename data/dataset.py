import glob, random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class HillshadeDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None,
                 road_biased=True, road_ratio=0.70, road_min_pixels=30):

        self.img_files  = sorted(glob.glob(f"{img_dir}/*.npy"))
        self.mask_files = sorted(glob.glob(f"{mask_dir}/*.npy"))
        self.transform  = transform

        self.road_files, self.bg_files = [], []

        if road_biased:
            for img_f, mask_f in zip(self.img_files, self.mask_files):
                mask = np.load(mask_f)
                if (mask.squeeze() > 0.5).sum() >= road_min_pixels:
                    self.road_files.append((img_f, mask_f))
                else:
                    self.bg_files.append((img_f, mask_f))

        self.road_biased = road_biased
        self.road_ratio  = road_ratio
        self._all = list(zip(self.img_files, self.mask_files))

    def __len__(self):
        return len(self.img_files)

    def _load(self, img_path, mask_path):
        img  = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        if img.shape[0] == 4:
            img = img.transpose(1, 2, 0)

        mask = (mask > 0.5).astype(np.float32)
        return img, mask

    def __getitem__(self, idx):
        if self.road_biased and self.road_files:
            pool = self.road_files if random.random() < self.road_ratio else self.bg_files
            img_p, mask_p = random.choice(pool)
        else:
            img_p, mask_p = self._all[idx]

        img, mask = self._load(img_p, mask_p)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        return img, mask.unsqueeze(0)