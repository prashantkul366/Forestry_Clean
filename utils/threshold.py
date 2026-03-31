import torch
import numpy as np
from engine.metrics import compute_metrics

@torch.no_grad()
def find_best_threshold(model, loader, cfg):
    model.eval()

    all_preds, all_targets = [], []

    for imgs, masks in loader:
        imgs = imgs.to(cfg.DEVICE)
        logits = model(imgs)

        all_preds.append(torch.sigmoid(logits).cpu())
        all_targets.append(masks.cpu())

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    best_t, best_dice = 0.5, 0.0

    for t in np.arange(0.10, 0.95, 0.05):
        m = compute_metrics((preds > t).float(), targets)

        if m["dice"] > best_dice:
            best_dice, best_t = m["dice"], float(t)

    return best_t, best_dice