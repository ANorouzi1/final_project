import torch
import torch.nn as nn
import torch.nn.functional as F


def total_variation_loss(prob, eps=1e-6):
    """Isotropic TV penalty over neighboring mask probabilities."""
    dx = prob[:, :, :-1, 1:] - prob[:, :, :-1, :-1]
    dy = prob[:, :, 1:, :-1] - prob[:, :, :-1, :-1]

    tv = torch.sqrt(dx.pow(2) + dy.pow(2) + eps)
    return tv.sum(dim=(-2, -1)).mean()


class DiceBCEDistanceTVLoss(nn.Module):
    """Compact dual-head loss: mask terms, SDF regression, and TV smoothing."""

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        distance_weight=0.5,
        tv_weight=0.02,
        eps=1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.distance_weight = distance_weight
        self.tv_weight = tv_weight
        self.eps = eps

    def forward(self, outputs, targets):
        mask_logits = outputs["mask_logits"]
        distance_logits = outputs["distance_logits"]
        target_mask = targets["mask"].float()
        target_distance = targets["distance"].float()

        bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask)
        mask_prob = torch.sigmoid(mask_logits)
        dice = self._dice_loss(mask_prob, target_mask)
        pred_distance = torch.tanh(distance_logits)
        distance = F.smooth_l1_loss(pred_distance, target_distance)
        tv = total_variation_loss(mask_prob, eps=self.eps)

        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.distance_weight * distance
            + self.tv_weight * tv
        )
        return {
            "loss": total,
            "bce": bce.detach(),
            "dice_loss": dice.detach(),
            "distance_loss": distance.detach(),
            "tv_loss": tv.detach(),
        }

    def _dice_loss(self, pred, target):
        pred = pred.flatten(1)
        target = target.flatten(1)
        intersection = (pred * target).sum(dim=1)
        denominator = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()


class DiceBCETVLoss(nn.Module):
    """Mask-only baseline loss with the same TV regularizer."""

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        tv_weight=0.02,
        eps=1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.tv_weight = tv_weight
        self.eps = eps

    def forward(self, outputs, targets):
        mask_logits = outputs["mask_logits"]
        target_mask = targets["mask"].float()

        bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask)
        mask_prob = torch.sigmoid(mask_logits)
        dice = self._dice_loss(mask_prob, target_mask)
        tv = total_variation_loss(mask_prob, eps=self.eps)
        distance = mask_logits.sum() * 0.0

        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.tv_weight * tv
        )
        return {
            "loss": total,
            "bce": bce.detach(),
            "dice_loss": dice.detach(),
            "distance_loss": distance.detach(),
            "tv_loss": tv.detach(),
        }

    def _dice_loss(self, pred, target):
        pred = pred.flatten(1)
        target = target.flatten(1)
        intersection = (pred * target).sum(dim=1)
        denominator = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()
