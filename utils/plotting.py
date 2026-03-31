def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # loss
    axes[0,0].plot(epochs, history["train_loss"], label="Train")
    axes[0,0].plot(epochs, history["val_loss"],   label="Val")
    axes[0,0].set_title("Loss"); axes[0,0].legend()

    # dice
    axes[0,1].plot(epochs, history["train_dice"], label="Train")
    axes[0,1].plot(epochs, history["val_dice"],   label="Val")
    axes[0,1].set_title("Dice"); axes[0,1].legend()

    # IoU
    axes[0,2].plot(epochs, history["train_iou"], label="Train")
    axes[0,2].plot(epochs, history["val_iou"],   label="Val")
    axes[0,2].set_title("IoU"); axes[0,2].legend()

    # precision / recall
    axes[1,0].plot(epochs, history["val_precision"],   label="Precision")
    axes[1,0].plot(epochs, history["val_recall"],      label="Recall")
    axes[1,0].set_title("Val Precision / Recall"); axes[1,0].legend()

    # specificity / accuracy
    axes[1,1].plot(epochs, history["val_specificity"], label="Specificity")
    axes[1,1].plot(epochs, history["val_accuracy"],    label="Accuracy")
    axes[1,1].set_title("Val Specificity / Accuracy"); axes[1,1].legend()

    # LR
    axes[1,2].plot(epochs, history["lr"])
    axes[1,2].set_title("Learning Rate")
    axes[1,2].set_yscale("log")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)

    plt.suptitle(f"{CFG.ARCHITECTURE}/{CFG.ENCODER}  —  Training History", fontsize=13)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=120)
    plt.show()
    print("Saved training_curves.png")


