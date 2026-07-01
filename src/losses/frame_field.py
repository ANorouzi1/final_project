"""Frame Field Learning loss (Girard et al., CVPR 2021), adapted for FTW fields.

This file is kept for reference/experimentation only. The active configs do not
import or use this loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def gt_tangent_from_sdf(sdf, band_sigma=0.12, eps=1e-6):
    """GT boundary tangent (unit complex, as real/imag) plus soft boundary weight."""
    pad = F.pad(sdf, (1, 1, 1, 1), mode="replicate")
    gx = (pad[..., 1:-1, 2:] - pad[..., 1:-1, :-2]) * 0.5
    gy = (pad[..., 2:, 1:-1] - pad[..., :-2, 1:-1]) * 0.5
    mag = torch.sqrt(gx * gx + gy * gy + eps)
    nx, ny = gx / mag, gy / mag
    taur, taui = -ny, nx
    weight = torch.exp(-(sdf / band_sigma) ** 2)
    return taur, taui, weight


def _grad_dir(field, eps):
    pad = F.pad(field, (1, 1, 1, 1), mode="replicate")
    gx = (pad[..., 1:-1, 2:] - pad[..., 1:-1, :-2]) * 0.5
    gy = (pad[..., 2:, 1:-1] - pad[..., :-2, 1:-1]) * 0.5
    mag = torch.sqrt(gx * gx + gy * gy + eps)
    return gx, gy, mag


class FrameFieldLoss(nn.Module):
    def __init__(
        self,
        align_weight=0.5,
        align90_weight=0.1,
        smooth_weight=0.02,
        seg_weight=0.2,
        eps=1e-6,
    ):
        super().__init__()
        self.align_weight = align_weight
        self.align90_weight = align90_weight
        self.smooth_weight = smooth_weight
        self.seg_weight = seg_weight
        self.eps = eps

    @staticmethod
    def _poly(z, c0, c2):
        z2 = z * z
        return z2 * z2 + c2 * z2 + c0

    def forward(self, frame_field, sdf, mask_logits=None):
        """Compute frame-field alignment, smoothness, and optional mask coupling."""
        with torch.no_grad():
            taur, taui, w = gt_tangent_from_sdf(sdf)
        c0 = torch.complex(frame_field[:, 0:1], frame_field[:, 1:2])
        c2 = torch.complex(frame_field[:, 2:3], frame_field[:, 3:4])
        tau = torch.complex(taur, taui)
        tau90 = torch.complex(-taui, taur)

        wsum = w.sum() + self.eps
        align = ((self._poly(tau, c0, c2).abs() ** 2) * w).sum() / wsum
        align90 = ((self._poly(tau90, c0, c2).abs() ** 2) * w).sum() / wsum

        dx = frame_field[:, :, :, 1:] - frame_field[:, :, :, :-1]
        dy = frame_field[:, :, 1:, :] - frame_field[:, :, :-1, :]
        smooth = (dx * dx).mean() + (dy * dy).mean()

        seg = torch.zeros((), device=frame_field.device)
        if self.seg_weight > 0 and mask_logits is not None:
            prob = torch.sigmoid(mask_logits)
            gx, gy, gmag = _grad_dir(prob, self.eps)
            t = torch.complex(-gy / gmag, gx / gmag)
            seg_pen = self._poly(t, c0.detach(), c2.detach()).abs() ** 2
            seg = (seg_pen * gmag).sum() / (gmag.sum() + self.eps)

        loss = (
            self.align_weight * align
            + self.align90_weight * align90
            + self.smooth_weight * smooth
            + self.seg_weight * seg
        )
        return {
            "loss": loss,
            "align": align.detach(),
            "align90": align90.detach(),
            "smooth": smooth.detach(),
            "seg": seg.detach(),
        }


class DiceBCEDistanceFrameFieldLoss(nn.Module):
    """Reference-only frame-field loss.

    The original pushed version wrapped an older geometric/polygon loss that is
    intentionally not part of the active training path. This class is kept so
    the file preserves the proposed frame-field objective, but configs should
    continue to use ``DiceBCEDistanceTVLoss`` unless this is deliberately wired
    up later.
    """

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        distance_weight=0.5,
        ff_align_weight=0.5,
        ff_align90_weight=0.1,
        ff_smooth_weight=0.02,
        ff_seg_weight=0.2,
        eps=1e-6,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.distance_weight = distance_weight
        self.eps = eps
        self.ff = FrameFieldLoss(
            ff_align_weight,
            ff_align90_weight,
            ff_smooth_weight,
            ff_seg_weight,
            eps,
        )

    def forward(self, outputs, targets):
        mask_logits = outputs["mask_logits"]
        target_mask = targets["mask"].float()
        pred_distance = torch.tanh(outputs["distance_logits"])
        target_distance = targets["distance"].float()

        bce = F.binary_cross_entropy_with_logits(mask_logits, target_mask)
        mask_prob = torch.sigmoid(mask_logits)
        dice = self._dice_loss(mask_prob, target_mask)
        distance = F.smooth_l1_loss(pred_distance, target_distance)
        ff = self.ff(
            outputs["frame_field"],
            target_distance,
            mask_logits=mask_logits,
        )

        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.distance_weight * distance
            + ff["loss"]
        )
        return {
            "loss": total,
            "bce": bce.detach(),
            "dice_loss": dice.detach(),
            "distance_loss": distance.detach(),
            "ff_align": ff["align"],
            "ff_align90": ff["align90"],
            "ff_smooth": ff["smooth"],
            "ff_seg": ff["seg"],
        }

    def _dice_loss(self, pred, target):
        pred = pred.flatten(1)
        target = target.flatten(1)
        intersection = (pred * target).sum(dim=1)
        denominator = pred.sum(dim=1) + target.sum(dim=1)
        dice = (2 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()
