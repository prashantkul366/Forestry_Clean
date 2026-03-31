import torch
# =============================================================================
#  CELL 8 — Train / Val loops
# =============================================================================
def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, masks in tqdm(loader, desc="  Train", leave=False):
        imgs, masks = imgs.to(CFG.DEVICE), masks.to(CFG.DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = loss_fn(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        with torch.no_grad():
            all_preds.append((torch.sigmoid(logits) > 0.5).float().cpu())
            all_targets.append(masks.cpu())

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    m = compute_metrics(preds, targets)
    return total_loss / len(loader), m


@torch.no_grad()
def validate(model, loader, loss_fn, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, masks in tqdm(loader, desc="  Val  ", leave=False):
        imgs, masks = imgs.to(CFG.DEVICE), masks.to(CFG.DEVICE)
        logits = model(imgs)
        total_loss += loss_fn(logits, masks).item()
        all_preds.append(torch.sigmoid(logits).cpu())
        all_targets.append(masks.cpu())

    preds   = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    m = compute_metrics((preds > threshold).float(), targets)
    return total_loss / len(loader), m


@torch.no_grad()
def find_best_threshold(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    for imgs, masks in loader:
        logits = model(imgs.to(CFG.DEVICE))
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



# def train(model, train_loader, val_loader, loss_fn):
def train(model, train_loader, val_loader, loss_fn, save_dir):

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=CFG.EPOCHS, eta_min=1e-6)

    history = {k: [] for k in [
        "train_loss", "val_loss",
        "train_dice", "train_iou", "train_precision",
        "train_recall", "train_specificity", "train_accuracy",
        "val_dice",   "val_iou",   "val_precision",
        "val_recall", "val_specificity", "val_accuracy",
        "lr"
    ]}

    best_dice     = 0.0
    best_epoch    = 0
    no_improve    = 0
    threshold     = 0.5

    # ── header ──────────────────────────────────────────────────────────────
    HDR = (f"\n{'Ep':>4} | {'LR':>8} | "
           f"{'TrLoss':>7} {'TrDice':>7} {'TrIoU':>6} {'TrPrec':>7} {'TrRec':>6} | "
           f"{'VaLoss':>7} {'VaDice':>7} {'VaIoU':>6} {'VaPrec':>7} {'VaRec':>6} "
           f"{'VaSpec':>7} {'VaAcc':>6} | {'Note'}")
    print(HDR)
    print("-" * len(HDR))

    for epoch in range(1, CFG.EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_m = train_one_epoch(model, train_loader, loss_fn, optimizer)
        va_loss, va_m = validate(model, val_loader, loss_fn, threshold)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]

        # tune threshold every 10 epochs
        if epoch % 10 == 0:
            threshold, _ = find_best_threshold(model, val_loader)

        # ── log history ───────────────────────────────────────────────────
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["lr"].append(lr_now)
        for k in ["dice","iou","precision","recall","specificity","accuracy"]:
            history[f"train_{k}"].append(tr_m[k])
            history[f"val_{k}"].append(va_m[k])

        # ── early stopping check ──────────────────────────────────────────
        note = ""
        if va_m["dice"] > best_dice:
            best_dice  = va_m["dice"]
            best_epoch = epoch
            no_improve = 0
            # ckpt_path  = f"{CFG.SAVE_DIR}/best_model.pth"
            ckpt_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                "epoch"    : epoch,
                "model"    : model.state_dict(),
                "dice"     : best_dice,
                "threshold": threshold,
                "cfg"      : {
                    "arch"      : CFG.ARCHITECTURE,
                    "encoder"   : CFG.ENCODER,
                    "in_channels": CFG.IN_CHANNELS,
                },
            }, ckpt_path)
            note = f"★ best  (thr={threshold:.2f})"
        else:
            no_improve += 1
            if no_improve >= CFG.PATIENCE:
                note = f"EARLY STOP (no improve {CFG.PATIENCE} epochs)"

        elapsed = time.time() - t0
        print(
            f"{epoch:4d} | {lr_now:.2e} | "
            f"{tr_loss:7.4f} {tr_m['dice']:7.4f} {tr_m['iou']:6.4f} "
            f"{tr_m['precision']:7.4f} {tr_m['recall']:6.4f} | "
            f"{va_loss:7.4f} {va_m['dice']:7.4f} {va_m['iou']:6.4f} "
            f"{va_m['precision']:7.4f} {va_m['recall']:6.4f} "
            f"{va_m['specificity']:7.4f} {va_m['accuracy']:6.4f} | "
            f"{note}  [{elapsed:.0f}s]"
        )

        if no_improve >= CFG.PATIENCE:
            print(f"\n  Stopped at epoch {epoch}. Best val Dice={best_dice:.4f} @ epoch {best_epoch}")
            break

    print(f"\n  Best val Dice : {best_dice:.4f}  @ epoch {best_epoch}")
    print(f"  Checkpoint    : {CFG.SAVE_DIR}/best_model.pth")
    return history, threshold


# history, best_threshold = train(model, train_loader, val_loader, loss_fn)