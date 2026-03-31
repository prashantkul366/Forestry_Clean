def compute_metrics(preds_bin, targets, eps=1e-6):
    p  = preds_bin.view(-1).float()
    t  = targets.view(-1).float()
    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()
    tn = ((1 - p) * (1 - t)).sum()

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * tp / (2 * tp + fp + fn + eps)
    iou       = tp / (tp + fp + fn + eps)
    specificity = tn / (tn + fp + eps)
    accuracy    = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "dice"       : f1.item(),
        "iou"        : iou.item(),
        "precision"  : precision.item(),
        "recall"     : recall.item(),
        "specificity": specificity.item(),
        "accuracy"   : accuracy.item(),
    }
