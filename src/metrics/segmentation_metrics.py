import torch
import torch.nn.functional as F

from src.utils.prediction import mask_probability_from_outputs


class MeanIoU:
    def __init__(
        self,
        threshold=0.5,
        eps=1e-6,
    ):
        self.threshold = threshold
        self.eps = eps

    def compute(self, outputs, targets):
        pred = mask_probability_from_outputs(outputs) > self.threshold
        target = targets["mask"] > 0.5
        intersection = (pred & target).flatten(1).sum(dim=1).float()
        union = (pred | target).flatten(1).sum(dim=1).float()
        return ((intersection + self.eps) / (union + self.eps)).mean().item()


class PixelIoU:
    """Global foreground pixel IoU, matching the paper-style pixel metric."""

    def __init__(
        self,
        threshold=0.5,
        eps=1e-6,
    ):
        self.threshold = threshold
        self.eps = eps

    def compute(self, outputs, targets):
        pred = mask_probability_from_outputs(outputs) > self.threshold
        target = targets["mask"] > 0.5
        intersection = (pred & target).sum().float()
        union = (pred | target).sum().float()
        return ((intersection + self.eps) / (union + self.eps)).item()


class BoundaryIoU:
    """IoU between symmetric boundary bands around prediction and target.

    ``radius`` controls how far each band extends both inward and outward from
    the corresponding mask boundary.
    """

    def __init__(
        self,
        threshold=0.5,
        radius=1,
        eps=1e-6,
    ):
        self.threshold = threshold
        self.radius = radius
        self.eps = eps

    def compute(self, outputs, targets):
        pred = (mask_probability_from_outputs(outputs) > self.threshold).float()
        target = (targets["mask"] > 0.5).float()
        pred_boundary = symmetric_boundary_band(pred, self.radius)
        target_boundary = symmetric_boundary_band(target, self.radius)
        intersection = (pred_boundary * target_boundary).flatten(1).sum(dim=1)
        union = ((pred_boundary + target_boundary) > 0).flatten(1).sum(dim=1).float()
        return ((intersection + self.eps) / (union + self.eps)).mean().item()


def symmetric_boundary_band(mask, radius):
    """Return a band extending ``radius`` pixels inside and outside a mask."""

    kernel = 2 * radius + 1
    mask = mask.float()
    dilated = F.max_pool2d(
        mask,
        kernel_size=kernel,
        stride=1,
        padding=radius,
    )
    eroded = -F.max_pool2d(
        -mask,
        kernel_size=kernel,
        stride=1,
        padding=radius,
    )
    return (dilated - eroded).clamp(min=0.0, max=1.0)
