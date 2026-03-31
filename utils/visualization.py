import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def visualize_predictions(model, val_loader, threshold, cfg, n=4, save_path=None):

    model.eval()
    imgs, masks = next(iter(val_loader))

    with torch.no_grad():
        imgs = imgs.to(cfg.DEVICE)
        logits = model(imgs)

        prob_maps = torch.sigmoid(logits).cpu()
        preds_bin = (prob_maps > threshold).float()

    fig, axes = plt.subplots(n, 5, figsize=(18, n * 4))

    for i in range(n):
        base = imgs[i, 0].cpu().numpy()
        base = (base - base.min()) / ((base.max() - base.min()) + 1e-6)

        gt   = masks[i, 0].numpy()
        prob = prob_maps[i, 0].numpy()
        pred = preds_bin[i, 0].numpy()

        

        inter = (pred * gt).sum()
        dice  = 2 * inter / (pred.sum() + gt.sum() + 1e-6)

        axes[i,0].imshow(base, cmap='gray')
        axes[i,0].set_title("Input C0")

        mosaic = np.concatenate(
            [(lambda c: (c - c.min()) / ((c.max() - c.min()) + 1e-6))(imgs[i, c].cpu().numpy())
             for c in range(4)], axis=1)
        axes[i,1].imshow(mosaic, cmap='gray')
        axes[i,1].set_title("All 4 channels")

        axes[i,2].imshow(gt, cmap='gray')
        axes[i,2].set_title("Ground truth")

        axes[i,3].imshow(prob, cmap='hot', vmin=0, vmax=1)
        axes[i,3].set_title(f"Prob (thr={threshold:.2f})")

        tp = (pred == 1) & (gt == 1)
        fp = (pred == 1) & (gt == 0)
        fn = (pred == 0) & (gt == 1)

        overlay = np.zeros((*pred.shape, 3))
        overlay[tp] = [0.0, 0.8, 0.0]
        overlay[fp] = [1.0, 0.5, 0.0]
        overlay[fn] = [1.0, 0.0, 0.0]

        axes[i,4].imshow(base, cmap='gray', alpha=0.6)
        axes[i,4].imshow(overlay, alpha=0.6)
        axes[i,4].set_title(f"TP/FP/FN Dice={dice:.3f}")

        for ax in axes[i]:
            ax.axis('off')

    patches = [
        mpatches.Patch(color=(0,0.8,0), label="True positive"),
        mpatches.Patch(color=(1,0.5,0), label="False positive"),
        mpatches.Patch(color=(1,0,0), label="False negative"),
    ]

    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=10)
    plt.suptitle("Prediction Analysis", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.savefig("prediction_analysis.png", dpi=120)
        print("Saved prediction_analysis.png")

    plt.show()