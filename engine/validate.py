
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