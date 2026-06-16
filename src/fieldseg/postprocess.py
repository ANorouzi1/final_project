from __future__ import annotations

from collections import deque

import numpy as np

from .distance import as_binary, boundary_from_mask
from .polygon import Point, simplify_polygon


def connected_components(mask: np.ndarray) -> np.ndarray:
    binary = as_binary(mask)
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(h):
        for x in range(w):
            if not binary[y, x] or labels[y, x] != 0:
                continue
            queue = deque([(y, x)])
            labels[y, x] = next_id
            while queue:
                cy, cx = queue.popleft()
                for dy, dx in directions:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = next_id
                        queue.append((ny, nx))
            next_id += 1
    return labels


def _local_maxima(distance: np.ndarray, mask: np.ndarray, min_distance: float) -> np.ndarray:
    dist = np.asarray(distance)
    binary = as_binary(mask)
    padded = np.pad(dist, 1, mode="constant", constant_values=-np.inf)
    maxima = binary & (dist >= min_distance)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            maxima &= dist >= padded[1 + dy : 1 + dy + dist.shape[0], 1 + dx : 1 + dx + dist.shape[1]]
    return connected_components(maxima)


def distance_seeded_instances(
    mask_prob: np.ndarray,
    distance: np.ndarray,
    mask_threshold: float = 0.5,
    seed_threshold: float = 0.35,
) -> np.ndarray:
    """Split a mask into instances using distance peaks as seeds."""
    mask = np.asarray(mask_prob) >= mask_threshold
    if not mask.any():
        return np.zeros_like(mask, dtype=np.int32)

    try:
        from scipy import ndimage as ndi
        from skimage.segmentation import watershed

        markers = _local_maxima(distance, mask, seed_threshold)
        if markers.max() == 0:
            markers = connected_components(mask)
        return watershed(-np.asarray(distance), markers=markers, mask=mask).astype(np.int32)
    except Exception:
        seeds = _local_maxima(distance, mask, seed_threshold)
        if seeds.max() == 0:
            return connected_components(mask)
        return _assign_to_nearest_seed(mask, seeds)


def _assign_to_nearest_seed(mask: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    labels = np.zeros_like(seeds, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()
    for y, x in zip(*np.nonzero(seeds)):
        labels[y, x] = seeds[y, x]
        queue.append((y, x))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        y, x = queue.popleft()
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                if mask[ny, nx] and labels[ny, nx] == 0:
                    labels[ny, nx] = labels[y, x]
                    queue.append((ny, nx))
    return labels


def contour_points(instance_mask: np.ndarray) -> list[Point]:
    boundary = boundary_from_mask(instance_mask)
    ys, xs = np.nonzero(boundary)
    if len(xs) == 0:
        return []
    cx = float(xs.mean())
    cy = float(ys.mean())
    points = sorted(zip(xs.astype(float), ys.astype(float)), key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
    return [(float(x), float(y)) for x, y in points]


def polygons_from_instances(instances: np.ndarray, epsilon: float = 2.0) -> dict[int, list[Point]]:
    polygons: dict[int, list[Point]] = {}
    for instance_id in np.unique(instances):
        if instance_id == 0:
            continue
        points = contour_points(instances == instance_id)
        if len(points) >= 3:
            polygons[int(instance_id)] = simplify_polygon(points, epsilon=epsilon)
    return polygons
