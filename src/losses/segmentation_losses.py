import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCEDistancePolygonLoss(nn.Module):
    """Mask loss plus an auxiliary signed distance field regression term."""

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        distance_weight=0.5,
        distance_positive_weight=0.0,
        sdf_gradient_weight=0.0,
        boundary_weight=0.0,
        boundary_radius=2,
        polygon_weight=0.05,
        distance_foreground_only=False,
        distance_mask_threshold=0.5,
        eps=1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.distance_weight = distance_weight
        self.distance_positive_weight = distance_positive_weight
        self.sdf_gradient_weight = sdf_gradient_weight
        self.boundary_weight = boundary_weight
        self.boundary_radius = boundary_radius
        self.polygon_weight = polygon_weight
        self.distance_foreground_only = distance_foreground_only
        self.distance_mask_threshold = distance_mask_threshold
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
        distance = self._distance_loss(pred_distance, target_distance, target_mask)
        sdf_gradient = self._sdf_gradient_loss(pred_distance, target_distance)
        boundary = self._boundary_loss(mask_logits, target_mask)
        polygon = self._polygon_smoothness(mask_prob)

        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.distance_weight * (distance + self.sdf_gradient_weight * sdf_gradient)
            + self.boundary_weight * boundary
            + self.polygon_weight * polygon
        )
        return {
            "loss": total,
            "bce": bce.detach(),
            "dice_loss": dice.detach(),
            "distance_loss": distance.detach(),
            "sdf_gradient_loss": sdf_gradient.detach(),
            "boundary_loss": boundary.detach(),
            "polygon_loss": polygon.detach(),
        }

    def _distance_loss(self, pred, target, mask):
        loss = F.smooth_l1_loss(pred, target, reduction="none")
        weights = torch.ones_like(target)
        if self.distance_positive_weight > 0:
            weights = weights + self.distance_positive_weight * target.clamp(min=0.0)
        if self.distance_foreground_only:
            weights = weights * (mask > self.distance_mask_threshold).float()
        return (loss * weights).sum() / weights.sum().clamp_min(self.eps)

    def _sdf_gradient_loss(self, pred, target):
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        return F.smooth_l1_loss(pred_dx, target_dx) + F.smooth_l1_loss(pred_dy, target_dy)

    def _boundary_loss(self, logits, target):
        if self.boundary_weight <= 0:
            return logits.sum() * 0.0
        boundary = self._boundary_band(target, self.boundary_radius)
        boundary_pixels = boundary.sum()
        if boundary_pixels <= 0:
            return logits.sum() * 0.0
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return (bce * boundary).sum() / boundary_pixels

    def _boundary_band(self, mask, radius):
        kernel = 2 * radius + 1
        dilated = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)
        eroded = -F.max_pool2d(-mask, kernel_size=kernel, stride=1, padding=radius)
        return (dilated - eroded).clamp(0.0, 1.0)

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


class DiceBCEPolygonLoss(nn.Module):
    """Mask-only baseline loss: same mask terms, no auxiliary distance target."""

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        polygon_weight=0.05,
        eps=1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.polygon_weight = polygon_weight
        self.eps = eps

    def forward(self, outputs, targets):
        mask_logits = outputs["mask_logits"]
        target_mask = targets["mask"].float()

        bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask)
        mask_prob = torch.sigmoid(mask_logits)
        dice = self._dice_loss(mask_prob, target_mask)
        polygon = self._polygon_smoothness(mask_prob)
        distance = mask_logits.sum() * 0.0
        sdf_gradient = mask_logits.sum() * 0.0
        boundary = mask_logits.sum() * 0.0

        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.polygon_weight * polygon
        )
        return {
            "loss": total,
            "bce": bce.detach(),
            "dice_loss": dice.detach(),
            "distance_loss": distance.detach(),
            "sdf_gradient_loss": sdf_gradient.detach(),
            "boundary_loss": boundary.detach(),
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
