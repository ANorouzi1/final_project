"""Frame Field Learning loss (Girard et al., CVPR 2021), adapted for FTW fields.

A frame field assigns to every pixel a pair of (generally non-orthogonal)
directions {u, -u, v, -v}, encoded by a degree-4 complex polynomial
    f(z) = z^4 + c2 z^2 + c0,   roots = {±u, ±v}.
The network predicts (c0, c2) as 4 real channels [Re c0, Im c0, Re c2, Im c2].

Why this fixes rounding/sharp corners (unlike a mask smoothness prior):
- a STRAIGHT edge => the boundary tangent is a constant direction => one frame
  root follows it (L_align);
- a CORNER => two different edge tangents meet; because u and v are independent,
  the 4-root polynomial can hold BOTH directions at once, so the corner is
  represented EXPLICITLY (a direction discontinuity), not smoothed away.

Unlike the axis-aligned TV "polygon" prior it replaces, the frame field is
rotation-EQUIVARIANT (no axis bias) and corner-preserving.

Terms (Girard):
  L_align   : f(tangent) = 0           -> a frame direction == boundary tangent
  L_align90 : f(i * tangent) = 0       -> clean cross on smooth edges
  L_smooth  : |grad c0|^2 + |grad c2|^2 -> spatial coherence (lets corners form)
  L_seg     : couple the predicted mask boundary -> frame field (sharpens MASK)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .segmentation_losses import DiceBCEDistancePolygonLoss


def gt_tangent_from_sdf(sdf, band_sigma=0.12, eps=1e-6):
    """GT boundary tangent (unit complex, as real/imag) + soft boundary weight.

    ``sdf`` is the per-instance signed distance target in ~[-1, 1] (0 on the
    field boundary). grad(sdf) points along the boundary normal; the tangent is
    that rotated by 90 degrees. Supervision is weighted to a soft band around the
    zero level set.
    """
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
    def __init__(self, align_weight=0.5, align90_weight=0.1, smooth_weight=0.02,
                 seg_weight=0.2, eps=1e-6):
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
        """frame_field: (B,4,H,W)=[Re c0,Im c0,Re c2,Im c2]; sdf: (B,1,H,W);
        mask_logits: (B,1,H,W) optional, enables the segmentation-coupling term."""
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

        loss = (self.align_weight * align
                + self.align90_weight * align90
                + self.smooth_weight * smooth
                + self.seg_weight * seg)
        return {"loss": loss, "align": align.detach(), "align90": align90.detach(),
                "smooth": smooth.detach(), "seg": seg.detach()}


class DiceBCEDistanceFrameFieldLoss(nn.Module):
    """Friend's mask/SDF loss (BCE + Dice + interior-weighted distance +
    SDF-gradient + boundary-weighted BCE) WITH the anisotropic polygon-TV term
    DISABLED, plus the Frame Field as the geometry prior."""

    def __init__(
        self,
        bce_weight=1.0,
        dice_weight=1.0,
        distance_weight=2.0,
        distance_positive_weight=3.0,
        sdf_gradient_weight=0.25,
        boundary_weight=0.30,
        boundary_radius=2,
        distance_foreground_only=False,
        ff_align_weight=0.5,
        ff_align90_weight=0.1,
        ff_smooth_weight=0.02,
        ff_seg_weight=0.2,
        eps=1e-6,
    ):
        super().__init__()
        self.base = DiceBCEDistancePolygonLoss(
            bce_weight=bce_weight, dice_weight=dice_weight, distance_weight=distance_weight,
            distance_positive_weight=distance_positive_weight, sdf_gradient_weight=sdf_gradient_weight,
            boundary_weight=boundary_weight, boundary_radius=boundary_radius,
            polygon_weight=0.0,  # anisotropic TV polygon prior OFF -> frame field replaces it
            distance_foreground_only=distance_foreground_only, eps=eps,
        )
        self.ff = FrameFieldLoss(ff_align_weight, ff_align90_weight, ff_smooth_weight,
                                 ff_seg_weight, eps)

    def forward(self, outputs, targets):
        d = self.base(outputs, targets)  # bce/dice/distance/sdf_gradient/boundary/polygon(=0)
        ff = self.ff(outputs["frame_field"], targets["distance"].float(),
                     mask_logits=outputs["mask_logits"])
        d["loss"] = d["loss"] + ff["loss"]
        d["ff_align"] = ff["align"]
        d["ff_align90"] = ff["align90"]
        d["ff_smooth"] = ff["smooth"]
        d["ff_seg"] = ff["seg"]
        return d
