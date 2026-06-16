import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label


class MeanIoU:
    def __init__(self, threshold=0.5, eps=1e-6):
        self.threshold = threshold
        self.eps = eps

    def compute(self, outputs, targets):
        pred = torch.sigmoid(outputs["mask_logits"]) > self.threshold
        target = targets["mask"] > 0.5
        intersection = (pred & target).flatten(1).sum(dim=1).float()
        union = (pred | target).flatten(1).sum(dim=1).float()
        return ((intersection + self.eps) / (union + self.eps)).mean().item()


class DiceScore:
    def __init__(self, threshold=0.5, eps=1e-6):
        self.threshold = threshold
        self.eps = eps

    def compute(self, outputs, targets):
        pred = torch.sigmoid(outputs["mask_logits"]) > self.threshold
        target = targets["mask"] > 0.5
        intersection = (pred & target).flatten(1).sum(dim=1).float()
        denominator = pred.flatten(1).sum(dim=1).float() + target.flatten(1).sum(dim=1).float()
        return ((2 * intersection + self.eps) / (denominator + self.eps)).mean().item()


class BoundaryIoU:
    def __init__(self, threshold=0.5, radius=2, eps=1e-6):
        self.threshold = threshold
        self.radius = radius
        self.eps = eps

    def compute(self, outputs, targets):
        pred = (torch.sigmoid(outputs["mask_logits"]) > self.threshold).float()
        target = (targets["mask"] > 0.5).float()
        pred_boundary = _mask_boundary(pred, self.radius)
        target_boundary = _mask_boundary(target, self.radius)
        intersection = (pred_boundary * target_boundary).flatten(1).sum(dim=1)
        union = ((pred_boundary + target_boundary) > 0).flatten(1).sum(dim=1).float()
        return ((intersection + self.eps) / (union + self.eps)).mean().item()


class InstanceF1:
    def __init__(self, threshold=0.5, iou_threshold=0.5):
        self.threshold = threshold
        self.iou_threshold = iou_threshold

    def compute(self, outputs, targets):
        pred_batch = (torch.sigmoid(outputs["mask_logits"]) > self.threshold).detach().cpu().numpy()
        if "instance" in targets:
            target_batch = targets["instance"].detach().cpu().numpy()
        else:
            target_batch = (targets["mask"] > 0.5).detach().cpu().numpy()
        scores = []
        for pred, target in zip(pred_batch, target_batch):
            pred_instances, _ = label(pred.squeeze() > 0)
            target_instances = target.squeeze()
            if target_instances.max() <= 1:
                target_instances, _ = label(target_instances > 0)
            tp, fp, fn, _ = _match_instances(pred_instances, target_instances, self.iou_threshold)
            denom = (2 * tp + fp + fn)
            scores.append(1.0 if denom == 0 else (2 * tp) / denom)
        return float(np.mean(scores))


class PanopticQualityApprox:
    def __init__(self, threshold=0.5, iou_threshold=0.5):
        self.threshold = threshold
        self.iou_threshold = iou_threshold

    def compute(self, outputs, targets):
        pred_batch = (torch.sigmoid(outputs["mask_logits"]) > self.threshold).detach().cpu().numpy()
        if "instance" in targets:
            target_batch = targets["instance"].detach().cpu().numpy()
        else:
            target_batch = (targets["mask"] > 0.5).detach().cpu().numpy()
        scores = []
        for pred, target in zip(pred_batch, target_batch):
            pred_instances, _ = label(pred.squeeze() > 0)
            target_instances = target.squeeze()
            if target_instances.max() <= 1:
                target_instances, _ = label(target_instances > 0)
            tp, fp, fn, iou_sum = _match_instances(pred_instances, target_instances, self.iou_threshold)
            denom = tp + 0.5 * fp + 0.5 * fn
            scores.append(1.0 if denom == 0 else iou_sum / denom)
        return float(np.mean(scores))


def _mask_boundary(mask, radius):
    pad = radius
    kernel = 2 * radius + 1
    eroded = -F.max_pool2d(-mask, kernel_size=kernel, stride=1, padding=pad)
    return (mask - eroded).clamp(min=0)


def _match_instances(pred, target, iou_threshold):
    pred_ids = [x for x in np.unique(pred) if x != 0]
    target_ids = [x for x in np.unique(target) if x != 0]
    pairs = []
    for pred_id in pred_ids:
        pred_mask = pred == pred_id
        for target_id in target_ids:
            target_mask = target == target_id
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            iou = 0.0 if union == 0 else intersection / union
            if iou >= iou_threshold:
                pairs.append((iou, pred_id, target_id))
    pairs.sort(reverse=True)

    matched_pred = set()
    matched_target = set()
    iou_sum = 0.0
    for iou, pred_id, target_id in pairs:
        if pred_id in matched_pred or target_id in matched_target:
            continue
        matched_pred.add(pred_id)
        matched_target.add(target_id)
        iou_sum += iou

    tp = len(matched_pred)
    fp = len(pred_ids) - tp
    fn = len(target_ids) - tp
    return tp, fp, fn, iou_sum
