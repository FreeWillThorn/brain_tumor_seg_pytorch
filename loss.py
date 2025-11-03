from mimetypes import inited

import torch
import torch.nn as nn
import torch.nn.functional as F



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

class DiceLoss_pic_wise(nn.Module):
    def __init__(self, smooth=1.0, reduction='none'):
        super(DiceLoss_pic_wise, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, preds, targets):
        preds = F.softmax(preds, dim=1)
        one_hot = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()

        intersection = (preds * one_hot).sum(dim=(2, 3))  # [B, C]
        union = preds.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))  # [B, C]
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice  # [B, C]
        return loss.mean(dim=1)  # [B] → per-image dice loss return average over


class HybridLoss(nn.Module):
    def __init__(self, ce_weight=None, dice_weight=0.9, ce_weight_tensor=None):
        super(HybridLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight_tensor)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.ce_weight = 1 - dice_weight

    def forward(self, preds, targets):
        hybrid_loss = self.ce_weight * self.ce(preds, targets) + self.dice_weight * self.dice(preds, targets)
        return hybrid_loss

class HybridLoss_image_wise_uncertainty(nn.Module):
    def __init__(self, ce_weight=None, dice_weight=0.9, ce_weight_tensor=None):
        super(HybridLoss_image_wise_uncertainty, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight_tensor)
        self.dice = DiceLoss_pic_wise()
        self.dice_weight = dice_weight
        self.ce_weight = 1 - dice_weight

    def forward(self, preds, targets,log_var):
        hybrid_loss = self.ce_weight * self.ce(preds, targets) + self.dice_weight * self.dice(preds, targets)
        #print(f"hybrid_loss shape: {hybrid_loss.shape}, hybrid_loss: {hybrid_loss}")
        # Scale by uncertainty
        log_var = log_var.squeeze(1)  # shape [B, H, W]
        log_var = F.adaptive_avg_pool2d(log_var, (1, 1)).squeeze()  # [B]
        #log_var = torch.clamp(log_var, -5.0, 5.0) # or use regularization, to prevent the model doesnt learn on hard area
        weighted_loss = (1 / (2 * torch.exp(log_var))) * hybrid_loss + 0.5 * log_var
        return weighted_loss.mean()


class HybridLoss_pixel_wise_uncertainty(nn.Module):
    def __init__(self, ce_weight_tensor=None, dice_weight=0.7):
        super(HybridLoss_pixel_wise_uncertainty, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight_tensor, reduction='none')
        self.dice = DiceLoss_pic_wise()  # per-image
        self.dice_weight = dice_weight
        self.ce_weight = 1 - dice_weight

    def forward(self, preds, targets, log_var):
        # preds: [B, C, H, W], targets: [B, H, W], log_var: [B, 1, H, W]
        #B = preds.shape[0]

        # Per-pixel CE
        ce_loss = self.ce(preds, targets)  # [B, H, W]
        log_var = log_var.squeeze(1)  # [B, H, W]
        log_var = torch.clamp(log_var, min=-5.0, max=5.0)
        ce_loss_scaled = (1 / (2 * torch.exp(log_var))) * ce_loss + 0.5 * log_var  # [B, H, W]
        ce_loss_scaled = ce_loss_scaled.mean(dim=(1, 2))  # [B]

        # Per-image Dice
        dice_loss = self.dice(preds, targets)  # [B]

        # Final hybrid loss per image
        hybrid_loss = self.ce_weight * ce_loss_scaled + self.dice_weight * dice_loss  # [B]
        if torch.isnan(hybrid_loss).any():
            print("NaN detected in hybrid_loss!")
            print("log_var stats:", log_var.min().item(), log_var.max().item())
            print("ce_loss_scaled stats:", ce_loss_scaled.min().item(), ce_loss_scaled.max().item())
            print("dice_loss stats:", dice_loss.min().item(), dice_loss.max().item())
            exit()

        return hybrid_loss.mean()  # scalar


def edge_dice_loss(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    intersection = (preds * targets).sum(1)
    union = preds.sum(1) + targets.sum(1)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def binary_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    probs = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_loss = alpha_t * (1 - p_t) ** gamma * ce_loss
    return focal_loss.mean()


def binary_dice_loss(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(1)
    union = preds_flat.sum(1) + targets_flat.sum(1)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

class combo_loss(nn.Module):
    def __init__(self, bce_weight=0.1):
        super(combo_loss, self).__init__()
        self.bce = None
        self.dice = None
        self.bce_weight = bce_weight

    def forward(self, preds, targets,scale=1.0):
        bce = F.binary_cross_entropy_with_logits(preds, targets, pos_weight=torch.tensor([20.0], device=preds.device))
        dice = binary_dice_loss(preds, targets)
        return scale * (self.bce_weight * bce + (1 - self.bce_weight) * dice)