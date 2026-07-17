import torch
import torch.nn as nn
import torch.nn.functional as F


def total_variation_loss(prob, eps=1e-6):
    """Isotropic TV penalty over neighboring mask probabilities."""
    dx = prob[:, :, :-1, 1:] - prob[:, :, :-1, :-1]
    dy = prob[:, :, 1:, :-1] - prob[:, :, :-1, :-1]

    tv = torch.sqrt(dx.pow(2) + dy.pow(2) + eps)
    return tv.sum(dim=(-2, -1)).mean()


def _dice_loss(pred, target, eps=1e-6):
    pred = pred.flatten(1)
    target = target.flatten(1)
    intersection = (pred * target).sum(dim=1)
    denominator = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def sdf_boundary_weight(target_distance, boundary_weight=3.0, boundary_sigma=0.12):
    """Higher pixel weights near SDF zero level sets."""
    boundary_sigma = max(float(boundary_sigma), 1e-6)
    return 1.0 + float(boundary_weight) * torch.exp(
        -target_distance.abs() / boundary_sigma
    )


def sdf_boundary_focus_weight(target_distance, boundary_sigma=0.12):
    """Soft focus mask for pixels near SDF zero level sets."""
    boundary_sigma = max(float(boundary_sigma), 1e-6)
    return torch.exp(-target_distance.abs() / boundary_sigma)


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
        target_mask = targets["mask"].float()

        bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask)
        mask_prob = torch.sigmoid(mask_logits)
        dice = _dice_loss(mask_prob, target_mask, eps=self.eps)
        if self.distance_weight:
            distance_logits = outputs["distance_logits"]
            target_distance = targets["distance"].float()
            pred_distance = torch.tanh(distance_logits)
            distance = F.smooth_l1_loss(pred_distance, target_distance)
        else:
            distance = mask_logits.new_zeros(())
        tv = total_variation_loss(mask_prob, eps=self.eps)

        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.distance_weight * distance
            + self.tv_weight * tv
        )
        weighted_distance = self.distance_weight * distance
        return {
            "loss": total,
            "bce": bce.detach(),
            "dice_loss": dice.detach(),
            "distance_loss": distance.detach(),
            "weighted_distance_loss": weighted_distance.detach(),
            "distance_loss_fraction": (
                weighted_distance / total.clamp_min(self.eps)
            ).detach(),
            "tv_loss": tv.detach(),
        }


class BoundaryWeightedSDFDiceBCEDistanceTVLoss(nn.Module):
    """Dual-head loss with normal mask terms and boundary-focused SDF regression."""

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        distance_weight=0.5,
        tv_weight=0.02,
        distance_boundary_weight=3.0,
        sdf_boundary_sigma=0.12,
        eps=1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.distance_weight = distance_weight
        self.tv_weight = tv_weight
        self.distance_boundary_weight = distance_boundary_weight
        self.sdf_boundary_sigma = sdf_boundary_sigma
        self.eps = eps

    def forward(self, outputs, targets):
        mask_logits = outputs["mask_logits"]
        target_mask = targets["mask"].float()

        bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask)
        mask_prob = torch.sigmoid(mask_logits)
        dice = _dice_loss(mask_prob, target_mask, eps=self.eps)

        if self.distance_weight:
            distance_logits = outputs["distance_logits"]
            target_distance = targets["distance"].float()
            pred_distance = torch.tanh(distance_logits)
            raw_distance = F.smooth_l1_loss(
                pred_distance,
                target_distance,
                reduction="none",
            )
            sdf_weights = sdf_boundary_weight(
                target_distance,
                boundary_weight=self.distance_boundary_weight,
                boundary_sigma=self.sdf_boundary_sigma,
            )
            distance = (
                (raw_distance * sdf_weights).sum()
                / sdf_weights.sum().clamp_min(self.eps)
            )
        else:
            sdf_weights = mask_logits.new_zeros(())
            distance = mask_logits.new_zeros(())
            raw_distance = distance
        tv = total_variation_loss(mask_prob, eps=self.eps)

        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.distance_weight * distance
            + self.tv_weight * tv
        )
        weighted_distance = self.distance_weight * distance
        return {
            "loss": total,
            "bce": bce.detach(),
            "dice_loss": dice.detach(),
            "distance_loss": distance.detach(),
            "raw_distance_loss": raw_distance.mean().detach(),
            "weighted_distance_loss": weighted_distance.detach(),
            "distance_loss_fraction": (
                weighted_distance / total.clamp_min(self.eps)
            ).detach(),
            "tv_loss": tv.detach(),
            "distance_boundary_weight_mean": sdf_weights.mean().detach(),
        }


class BoundaryWeightedDiceBCEDistanceTVLoss(nn.Module):
    """Dual-head loss with BCE focused near field boundaries."""

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        distance_weight=0.5,
        tv_weight=0.02,
        boundary_weight=3.0,
        boundary_sigma=0.12,
        eps=1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.distance_weight = distance_weight
        self.tv_weight = tv_weight
        self.boundary_weight = boundary_weight
        self.boundary_sigma = boundary_sigma
        self.eps = eps

    def forward(self, outputs, targets):
        mask_logits = outputs["mask_logits"]
        target_mask = targets["mask"].float()
        target_distance = targets["distance"].float()

        raw_bce = F.binary_cross_entropy_with_logits(
            mask_logits,
            target_mask,
            reduction="none",
        )
        bce_weights = sdf_boundary_weight(
            target_distance,
            boundary_weight=self.boundary_weight,
            boundary_sigma=self.boundary_sigma,
        )
        bce = (raw_bce * bce_weights).sum() / bce_weights.sum().clamp_min(self.eps)

        mask_prob = torch.sigmoid(mask_logits)
        dice = _dice_loss(mask_prob, target_mask, eps=self.eps)
        if self.distance_weight:
            distance_logits = outputs["distance_logits"]
            pred_distance = torch.tanh(distance_logits)
            distance = F.smooth_l1_loss(pred_distance, target_distance)
        else:
            distance = mask_logits.new_zeros(())
        tv = total_variation_loss(mask_prob, eps=self.eps)

        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.distance_weight * distance
            + self.tv_weight * tv
        )
        weighted_distance = self.distance_weight * distance
        return {
            "loss": total,
            "bce": bce.detach(),
            "raw_bce": raw_bce.mean().detach(),
            "dice_loss": dice.detach(),
            "distance_loss": distance.detach(),
            "weighted_distance_loss": weighted_distance.detach(),
            "distance_loss_fraction": (
                weighted_distance / total.clamp_min(self.eps)
            ).detach(),
            "tv_loss": tv.detach(),
            "boundary_weight_mean": bce_weights.mean().detach(),
        }


class BoundaryWeightedDiceBCEBoundarySDFTVLoss(nn.Module):
    """Dual-head loss with boundary-focused BCE and SDF regression."""

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        distance_weight=0.5,
        tv_weight=0.02,
        boundary_weight=3.0,
        boundary_sigma=0.12,
        distance_boundary_weight=None,
        sdf_boundary_sigma=None,
        eps=1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.distance_weight = distance_weight
        self.tv_weight = tv_weight
        self.boundary_weight = boundary_weight
        self.boundary_sigma = boundary_sigma
        self.distance_boundary_weight = (
            boundary_weight if distance_boundary_weight is None else distance_boundary_weight
        )
        self.sdf_boundary_sigma = (
            boundary_sigma if sdf_boundary_sigma is None else sdf_boundary_sigma
        )
        self.eps = eps

    def forward(self, outputs, targets):
        mask_logits = outputs["mask_logits"]
        target_mask = targets["mask"].float()
        target_distance = targets["distance"].float()

        raw_bce = F.binary_cross_entropy_with_logits(
            mask_logits,
            target_mask,
            reduction="none",
        )
        bce_weights = sdf_boundary_weight(
            target_distance,
            boundary_weight=self.boundary_weight,
            boundary_sigma=self.boundary_sigma,
        )
        bce = (raw_bce * bce_weights).sum() / bce_weights.sum().clamp_min(self.eps)

        mask_prob = torch.sigmoid(mask_logits)
        dice = _dice_loss(mask_prob, target_mask, eps=self.eps)

        if self.distance_weight:
            distance_logits = outputs["distance_logits"]
            pred_distance = torch.tanh(distance_logits)
            raw_distance = F.smooth_l1_loss(
                pred_distance,
                target_distance,
                reduction="none",
            )
            distance_weights = sdf_boundary_weight(
                target_distance,
                boundary_weight=self.distance_boundary_weight,
                boundary_sigma=self.sdf_boundary_sigma,
            )
            distance = (
                (raw_distance * distance_weights).sum()
                / distance_weights.sum().clamp_min(self.eps)
            )
        else:
            distance = mask_logits.new_zeros(())
            raw_distance = distance
            distance_weights = mask_logits.new_zeros(())

        tv = total_variation_loss(mask_prob, eps=self.eps)

        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.distance_weight * distance
            + self.tv_weight * tv
        )
        weighted_distance = self.distance_weight * distance
        return {
            "loss": total,
            "bce": bce.detach(),
            "raw_bce": raw_bce.mean().detach(),
            "dice_loss": dice.detach(),
            "distance_loss": distance.detach(),
            "raw_distance_loss": raw_distance.mean().detach(),
            "weighted_distance_loss": weighted_distance.detach(),
            "distance_loss_fraction": (
                weighted_distance / total.clamp_min(self.eps)
            ).detach(),
            "tv_loss": tv.detach(),
            "boundary_weight_mean": bce_weights.mean().detach(),
            "distance_boundary_weight_mean": distance_weights.mean().detach(),
        }


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
        dice = _dice_loss(mask_prob, target_mask, eps=self.eps)
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


class DiceBCESeamLoss(nn.Module):
    """Mask-only loss with an instance-derived seam weight map.

    This keeps the same BCE + Dice (+ TV) structure as ``DiceBCETVLoss``, but
    weights per-pixel BCE by ``targets["seam_weight"]``. The dataset builds that
    map from the instance labels so pixels between neighboring fields can matter
    more during mask training.
    """

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
        seam_weight = targets["seam_weight"].float()

        raw_bce = F.binary_cross_entropy_with_logits(
            mask_logits,
            target_mask,
            reduction="none",
        )
        bce = (raw_bce * seam_weight).sum() / seam_weight.sum().clamp_min(self.eps)
        mask_prob = torch.sigmoid(mask_logits)
        dice = _dice_loss(mask_prob, target_mask, eps=self.eps)
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
            "seam_weight_mean": seam_weight.mean().detach(),
        }
