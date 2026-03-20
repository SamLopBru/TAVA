import torch.nn as nn
from metrics.metrics import dice_score

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        return 1.0 - dice_score(logits, targets, smooth=self.smooth)


class CombinedLoss(nn.Module):
    """
    Combines BCEWithLogitsLoss and DiceLoss.
    Optimizes for pixel-wise accuracy (BCE) and regional overlap (Dice) simultaneously.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()  # Internal computation handles applies sigmoid
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss