from configs.config import CFG
from data.dataset import HillshadeDataset
from data.transforms import get_transforms
from models.model import build_model
from losses.losses import CombinedLoss
from engine.train import train

from torch.utils.data import DataLoader

def main():

    print("\nBuilding datasets ...")
    train_ds = HillshadeDataset(CFG.TRAIN_IMGS, CFG.TRAIN_MASKS,
                                transform=get_transforms("train"),
                                road_biased=True,
                                road_ratio=CFG.ROAD_RATIO,
                                road_min_pixels=CFG.ROAD_MIN_PIXELS)
    val_ds   = HillshadeDataset(CFG.VAL_IMGS, CFG.VAL_MASKS,
                                transform=get_transforms("val"),
                                road_biased=False)

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE,
                            shuffle=True,  num_workers=CFG.NUM_WORKERS,
                            pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG.BATCH_SIZE,
                            shuffle=False, num_workers=CFG.NUM_WORKERS,
                            pin_memory=True)

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model : {CFG.ARCHITECTURE} / {CFG.ENCODER}")
    print(f"Params: {n_params/1e6:.1f}M  on {CFG.DEVICE}")
    
    loss_fn = CombinedLoss(CFG.POS_WEIGHT, CFG.DICE_WEIGHT, CFG.BCE_WEIGHT, CFG.FOCAL_GAMMA)
    print(f"\nLoss ready — pos_weight={CFG.POS_WEIGHT}")

    history, thr = train(model, train_loader, val_loader, loss_fn)

if __name__ == "__main__":
    main()