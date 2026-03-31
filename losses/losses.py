
# =============================================================================
#  CELL 5 — Loss
# =============================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits).view(-1)
        t = targets.view(-1)
        inter = (p * t).sum()
        return 1.0 - (2.0 * inter + self.smooth) / \
                     (p.sum() + t.sum() + self.smooth)


class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        pw  = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        bce = F.binary_cross_entropy_with_logits(
                  logits, targets, pos_weight=pw, reduction='none')
        pt    = torch.exp(-bce)
        focal = ((1 - pt) ** self.gamma) * bce
        return focal.mean()


class CombinedLoss(nn.Module):
    def __init__(self, pos_weight, dice_w=0.5, bce_w=0.5, gamma=2.0):
        super().__init__()
        self.dice  = DiceLoss()
        self.focal = FocalBCELoss(gamma=gamma,
                                  pos_weight=torch.tensor([pos_weight]))
        self.dw = dice_w
        self.bw = bce_w

    def forward(self, logits, targets):
        return self.dw * self.dice(logits, targets) + \
               self.bw * self.focal(logits, targets)

