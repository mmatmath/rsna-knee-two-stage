import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        logits = logits.reshape(-1)
        targets = targets.reshape(-1)
        known = targets != self.ignore_index
        logits = logits[known]
        targets = targets[known]

        if logits.numel() == 0:
            return logits.sum()

        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probability = torch.sigmoid(logits)
        pt = probability * targets + (1.0 - probability) * (1.0 - targets)
        loss = loss * (1.0 - pt).pow(self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = loss * alpha_t

        return loss.mean()


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        known = targets != self.ignore_index
        if not known.any():
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(logits[known], targets[known])
