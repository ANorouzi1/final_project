from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    intersection = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denom + eps)).mean()


def _gradient(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dy = tensor[..., 1:, :] - tensor[..., :-1, :]
    dx = tensor[..., :, 1:] - tensor[..., :, :-1]
    return dy, dx


def polygon_regularization(segmentation_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Encourage clean boundaries with low spurious curvature near field edges."""
    prob = torch.sigmoid(segmentation_logits)
    dy, dx = _gradient(prob)
    ddy = dy[..., 1:, :] - dy[..., :-1, :]
    ddx = dx[..., :, 1:] - dx[..., :, :-1]

    target_dy, target_dx = _gradient(target)
    boundary_y = F.pad(target_dy.abs(), (0, 0, 0, 1))
    boundary_x = F.pad(target_dx.abs(), (0, 1, 0, 0))
    boundary_weight = torch.clamp(boundary_y + boundary_x, 0.0, 1.0)

    curvature = F.pad(ddy.abs(), (0, 0, 1, 1)) + F.pad(ddx.abs(), (1, 1, 0, 0))
    return (curvature * (0.25 + boundary_weight)).mean()


def combined_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    distance_weight: float = 0.4,
    polygon_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["segmentation_logits"]
    distance = outputs["distance"]
    seg_target = batch["segmentation"]
    dist_target = batch["distance"]

    bce = F.binary_cross_entropy_with_logits(logits, seg_target)
    dice = dice_loss(logits, seg_target)
    distance_loss = F.smooth_l1_loss(distance, dist_target)
    poly = polygon_regularization(logits, seg_target)
    total = bce + dice + distance_weight * distance_loss + polygon_weight * poly
    return total, {
        "bce": float(bce.detach().cpu()),
        "dice": float(dice.detach().cpu()),
        "distance": float(distance_loss.detach().cpu()),
        "polygon": float(poly.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
