import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        preds: [B, C, H, W] raw logits
        targets: [B, H, W] ground truth class indices
        """
        preds = F.softmax(preds, dim=1)  # Convert logits to probabilities

        one_hot = F.one_hot(targets, num_classes=preds.shape[1])  # [B, H, W, C]
        one_hot = one_hot.permute(0, 3, 1, 2).float()              # [B, C, H, W]

        intersection = (preds * one_hot).sum(dim=(2, 3))           # [B, C]
        union = preds.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))    # [B, C]

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        return loss.mean()  # Mean over batch and classes

class HybridLoss(nn.Module):
    def __init__(self, ce_weight=None, dice_weight=0.5, ce_weight_tensor=None):
        super(HybridLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight_tensor)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.ce_weight = 1 - dice_weight

    def forward(self, preds, targets):
        return self.ce_weight * self.ce(preds, targets) + self.dice_weight * self.dice(preds, targets)


class EdgeFocalLoss:
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, input, target):
        return sigmoid_focal_loss(
            input,
            target,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction
        )


class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        if isinstance(alpha, (list, torch.Tensor)):
            self.alpha = alpha.detach().clone() if isinstance(alpha, torch.Tensor) else torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None  # scalar alpha will be applied uniformly
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: [B, C, H, W] — raw logits
        targets: [B, H, W] — integer class labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # [B, H, W]
        pt = torch.exp(-ce_loss)  # softmax probability of correct class

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            # Gather alpha for each target pixel
            alpha_t = self.alpha[targets]  # [B, H, W]
        else:
            alpha_t = 1.0

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss