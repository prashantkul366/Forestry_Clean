import sys
import os


# add project root to python path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from utils.seed import set_seed
# from configs.config import CFG
# from data.dataset import HillshadeDataset
# from data.transforms import get_transforms
# from models.model import build_model
# from losses.losses import CombinedLoss
# from engine.train import train

# from torch.utils.data import DataLoader

# def main():

#     print(f"Device : {CFG.DEVICE}")
#     print(f"PyTorch: {torch.__version__}")
 
#     print("\nBuilding datasets ...")
#     train_ds = HillshadeDataset(CFG.TRAIN_IMGS, CFG.TRAIN_MASKS,
#                                 transform=get_transforms("train"),
#                                 road_biased=True,
#                                 road_ratio=CFG.ROAD_RATIO,
#                                 road_min_pixels=CFG.ROAD_MIN_PIXELS)
#     val_ds   = HillshadeDataset(CFG.VAL_IMGS, CFG.VAL_MASKS,
#                                 transform=get_transforms("val"),
#                                 road_biased=False)

#     train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE,
#                             shuffle=True,  num_workers=CFG.NUM_WORKERS,
#                             pin_memory=True)
#     val_loader   = DataLoader(val_ds,   batch_size=CFG.BATCH_SIZE,
#                             shuffle=False, num_workers=CFG.NUM_WORKERS,
#                             pin_memory=True)

#     imgs, masks = next(iter(train_loader))
#     print(f"\nBatch img  : {imgs.shape}  range=[{imgs.min():.3f}, {imgs.max():.3f}]")
#     print(f"Batch mask : {masks.shape}  unique={masks.unique().tolist()}")
    
#     model = build_model()
#     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Model : {CFG.ARCHITECTURE} / {CFG.ENCODER}")
#     print(f"Params: {n_params/1e6:.1f}M  on {CFG.DEVICE}")
    
#     loss_fn = CombinedLoss(CFG.POS_WEIGHT, CFG.DICE_WEIGHT, CFG.BCE_WEIGHT, CFG.FOCAL_GAMMA)
#     print(f"\nLoss ready — pos_weight={CFG.POS_WEIGHT}")

#     history, thr = train(model, train_loader, val_loader, loss_fn)
#     plot_history(history)
#     final_metrics = final_eval(model, val_loader)
#     visualize_predictions(model, val_loader, final_metrics["thresh"])
 

# if __name__ == "__main__":
#     main()

import os
import json
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from configs.config import CFG
from data.dataset import HillshadeDataset
from data.transforms import get_transforms
from models.model import build_model
from losses.losses import CombinedLoss

from engine.train import train
from utils.plotting import plot_history
from utils.visualization import visualize_predictions
from utils.threshold import find_best_threshold
from engine.metrics import compute_metrics


# ============================================================
# 🔹 Experiment Setup
# ============================================================
def setup_experiment():
    from datetime import datetime
    import os, json

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_name = f"{CFG.ARCHITECTURE}_{CFG.ENCODER}_{timestamp}"
    save_dir = os.path.join(CFG.SAVE_DIR, exp_name)

    os.makedirs(save_dir, exist_ok=True)

    # ✅ FIX HERE
    cfg_dict = {
        k: v for k, v in CFG.__dict__.items()
        if not k.startswith("__") and not callable(v)
    }

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=4)

    print(f"\n📁 Experiment: {exp_name}")
    print(f"📁 Save dir  : {save_dir}")

    return save_dir


# ============================================================
# 🔹 Data
# ============================================================
def build_dataloaders():

    from utils.seed import seed_worker

    g = torch.Generator()
    g.manual_seed(CFG.SEED)
    print("\n📦 Building datasets...")

    print("TRAIN IMG DIR:", CFG.TRAIN_IMGS)
    print("TRAIN MASK DIR:", CFG.TRAIN_MASKS)

    import glob
    print("Num train images:", len(glob.glob(f"{CFG.TRAIN_IMGS}/*.npy")))
    print("Num train masks :", len(glob.glob(f"{CFG.TRAIN_MASKS}/*.npy")))

    train_ds = HillshadeDataset(
        CFG.TRAIN_IMGS,
        CFG.TRAIN_MASKS,
        transform=get_transforms("train"),
        road_biased=True,
        road_ratio=CFG.ROAD_RATIO,
        road_min_pixels=CFG.ROAD_MIN_PIXELS,
    )

    val_ds = HillshadeDataset(
        CFG.VAL_IMGS,
        CFG.VAL_MASKS,
        transform=get_transforms("val"),
        road_biased=False,
    )

    # train_loader = DataLoader(
    #     train_ds,
    #     batch_size=CFG.BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=CFG.NUM_WORKERS,
    #     pin_memory=True,
    # )
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=CFG.BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=CFG.NUM_WORKERS,
    #     pin_memory=True,
    # )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.BATCH_SIZE,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader



# ============================================================
# 🔹 Model + Loss
# ============================================================
def build_model_and_loss():
    model = build_model()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model : {CFG.ARCHITECTURE} / {CFG.ENCODER}")
    print(f"🔢 Params: {n_params/1e6:.1f}M")

    loss_fn = CombinedLoss(
        CFG.POS_WEIGHT,
        CFG.DICE_WEIGHT,
        CFG.BCE_WEIGHT,
        CFG.FOCAL_GAMMA,
    )

    return model, loss_fn


# ============================================================
# 🔹 Save history
# ============================================================
def save_history(history, save_dir):
    path = os.path.join(save_dir, "history.json")
    with open(path, "w") as f:
        json.dump(history, f, indent=4)


# ============================================================
# 🔹 Final Evaluation
# ============================================================
@torch.no_grad()
def final_evaluation(model, val_loader, save_dir):
    print("\n📊 Running final evaluation...")

    all_preds, all_targets = [], []

    model.eval()
    for imgs, masks in val_loader:
        imgs = imgs.to(CFG.DEVICE)
        logits = model(imgs)

        all_preds.append(torch.sigmoid(logits).cpu())
        all_targets.append(masks)

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    best_t, _ = find_best_threshold(model, val_loader)

    metrics = compute_metrics((preds > best_t).float(), targets)

    print("\n✅ Final Metrics:")
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.4f}")

    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    return best_t, metrics


# ============================================================
# 🔹 MAIN
# ============================================================
def main():

    set_seed(CFG.SEED)
    start_time = time.time()

    # 1. setup experiment
    save_dir = setup_experiment()

    # 2. data
    train_loader, val_loader = build_dataloaders()

    imgs, masks = next(iter(train_loader))
    print(f"\nBatch img  : {imgs.shape}  range=[{imgs.min():.3f}, {imgs.max():.3f}]")
    print(f"Batch mask : {masks.shape}  unique={masks.unique().tolist()}")
    
    # 3. model + loss
    model, loss_fn = build_model_and_loss()

    # 4. train
    history, best_thr = train(
        model,
        train_loader,
        val_loader,
        loss_fn,
        cfg=CFG,
        save_dir=save_dir
    )

    # 5. save history
    save_history(history, save_dir)

    # 6. plots
    # plot_history(history, save_path=os.path.join(save_dir, "training_curves.png"))
    plot_history(
        history,
        cfg=CFG,
        save_path=os.path.join(save_dir, "training_curves.png")
    )

    # 7. final eval
    best_thr, metrics = final_evaluation(model, val_loader, save_dir)

    # 8. visualization
    visualize_predictions(
        model,
        val_loader,
        threshold=best_thr,
        save_path=os.path.join(save_dir, "predictions.png"),
    )

    print(f"\n⏱ Total time: {(time.time() - start_time)/60:.2f} min")
    print(f"📁 Outputs saved in: {save_dir}")


if __name__ == "__main__":
    main()