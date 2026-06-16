from __future__ import annotations

import numpy as np

from .distance import as_binary, boundary_from_mask


def mean_iou(pred_mask: np.ndarray, true_mask: np.ndarray, threshold: float = 0.5) -> float:
    pred = np.asarray(pred_mask) >= threshold
    true = as_binary(true_mask)
    intersection = np.logical_and(pred, true).sum()
    union = np.logical_or(pred, true).sum()
    return float(intersection / union) if union else 1.0


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return as_binary(mask)
    binary = as_binary(mask)
    padded = np.pad(binary, radius, mode="constant", constant_values=False)
    out = np.zeros_like(binary, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                y0 = radius + dy
                x0 = radius + dx
                out |= padded[y0 : y0 + binary.shape[0], x0 : x0 + binary.shape[1]]
    return out


def boundary_iou(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    threshold: float = 0.5,
    dilation_radius: int = 2,
) -> float:
    pred_boundary = _dilate(boundary_from_mask(np.asarray(pred_mask) >= threshold), dilation_radius)
    true_boundary = _dilate(boundary_from_mask(true_mask), dilation_radius)
    intersection = np.logical_and(pred_boundary, true_boundary).sum()
    union = np.logical_or(pred_boundary, true_boundary).sum()
    return float(intersection / union) if union else 1.0


def _instance_ids(mask: np.ndarray) -> list[int]:
    return [int(i) for i in np.unique(mask) if int(i) != 0]


def _pairwise_iou(pred_instances: np.ndarray, true_instances: np.ndarray) -> list[tuple[float, int, int]]:
    pairs = []
    for pred_id in _instance_ids(pred_instances):
        pred = pred_instances == pred_id
        for true_id in _instance_ids(true_instances):
            true = true_instances == true_id
            union = np.logical_or(pred, true).sum()
            if union:
                pairs.append((float(np.logical_and(pred, true).sum() / union), pred_id, true_id))
    return sorted(pairs, reverse=True)


def match_instances(
    pred_instances: np.ndarray,
    true_instances: np.ndarray,
    iou_threshold: float = 0.5,
) -> tuple[list[float], int, int]:
    matched_pred: set[int] = set()
    matched_true: set[int] = set()
    matches: list[float] = []

    for iou, pred_id, true_id in _pairwise_iou(pred_instances, true_instances):
        if iou < iou_threshold:
            continue
        if pred_id in matched_pred or true_id in matched_true:
            continue
        matched_pred.add(pred_id)
        matched_true.add(true_id)
        matches.append(iou)

    false_positive = len(_instance_ids(pred_instances)) - len(matched_pred)
    false_negative = len(_instance_ids(true_instances)) - len(matched_true)
    return matches, false_positive, false_negative


def instance_f1(
    pred_instances: np.ndarray,
    true_instances: np.ndarray,
    iou_threshold: float = 0.5,
) -> float:
    matches, fp, fn = match_instances(pred_instances, true_instances, iou_threshold)
    tp = len(matches)
    denom = 2 * tp + fp + fn
    return float((2 * tp) / denom) if denom else 1.0


def panoptic_quality(
    pred_instances: np.ndarray,
    true_instances: np.ndarray,
    iou_threshold: float = 0.5,
) -> float:
    matches, fp, fn = match_instances(pred_instances, true_instances, iou_threshold)
    denom = len(matches) + 0.5 * fp + 0.5 * fn
    return float(sum(matches) / denom) if denom else 1.0
