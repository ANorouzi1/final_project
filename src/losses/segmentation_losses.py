import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCEDistancePolygonLoss(nn.Module):
    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        distance_weight=0.5,
        polygon_weight=0.05,
        eps=1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.distance_weight = distance_weight
        self.polygon_weight = polygon_weight
        self.eps = eps

    def forward(self, outputs, targets):
        mask_logits = outputs["mask_logits"]
        distance_logits = outputs["distance_logits"]
        target_mask = targets["mask"].float()
        target_distance = targets["distance"].float()

        bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask)
        mask_prob = torch.sigmoid(mask_logits)
        dice = self._dice_loss(mask_prob, target_mask)
        distance = F.smooth_l1_loss(torch.sigmoid(distance_logits), target_distance)
        polygon = self._polygon_smoothness(mask_prob)

        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.distance_weight * distance
            + self.polygon_weight * polygon
        )
        return {
            "loss": total,
            "bce": bce.detach(),
            "dice_loss": dice.detach(),
            "distance_loss": distance.detach(),
            "polygon_loss": polygon.detach(),
        }

    def _dice_loss(self, pred, target):
        pred = pred.flatten(1)
        target = target.flatten(1)
        intersection = (pred * target).sum(dim=1)
        denominator = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()

    def _polygon_smoothness(self, prob):
        dx = prob[:, :, :, 1:] - prob[:, :, :, :-1]
        dy = prob[:, :, 1:, :] - prob[:, :, :-1, :]
        ddx = dx[:, :, :, 1:] - dx[:, :, :, :-1]
        ddy = dy[:, :, 1:, :] - dy[:, :, :-1, :]
        return ddx.abs().mean() + ddy.abs().mean()
