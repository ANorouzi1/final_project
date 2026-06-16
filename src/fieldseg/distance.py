from __future__ import annotations

from collections import deque

import numpy as np


def as_binary(mask: np.ndarray) -> np.ndarray:
    """Return a 2D boolean mask where non-zero pixels are foreground."""
    if mask.ndim == 3:
        mask = mask[..., 0]
    return np.asarray(mask) > 0


def boundary_from_mask(mask: np.ndarray) -> np.ndarray:
    """Find foreground pixels touching background or the image border."""
    fg = as_binary(mask)
    if fg.size == 0:
        return fg

    padded = np.pad(fg, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    neighbors = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[:-2, :-2]
        & padded[:-2, 2:]
        & padded[2:, :-2]
        & padded[2:, 2:]
    )
    interior = center & neighbors
    return fg & ~interior


def _bfs_distance_to_background(mask: np.ndarray) -> np.ndarray:
    """Fallback chamfer distance when scipy is unavailable."""
    fg = as_binary(mask)
    h, w = fg.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    queue: deque[tuple[int, int]] = deque()

    for y in range(h):
        for x in range(w):
            if not fg[y, x]:
                dist[y, x] = 0.0
                queue.append((y, x))

    if not queue:
        return np.ones((h, w), dtype=np.float32)

    directions = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 1.4142),
        (-1, 1, 1.4142),
        (1, -1, 1.4142),
        (1, 1, 1.4142),
    ]

    while queue:
        y, x = queue.popleft()
        base = dist[y, x]
        for dy, dx, weight in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and base + weight < dist[ny, nx]:
                dist[ny, nx] = base + weight
                queue.append((ny, nx))
    return dist


def distance_transform(mask: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Distance from each foreground pixel to nearest background pixel."""
    fg = as_binary(mask)
    try:
        from scipy.ndimage import distance_transform_edt

        dist = distance_transform_edt(fg).astype(np.float32)
    except Exception:
        dist = _bfs_distance_to_background(fg)
        dist[~fg] = 0.0

    if normalize and dist.max() > 0:
        dist = dist / dist.max()
    return dist.astype(np.float32)


def make_targets(mask: np.ndarray) -> dict[str, np.ndarray]:
    """Create segmentation and distance targets from a binary or instance mask."""
    seg = as_binary(mask).astype(np.float32)
    dist = distance_transform(seg, normalize=True)
    boundary = boundary_from_mask(seg).astype(np.float32)
    return {"segmentation": seg, "distance": dist, "boundary": boundary}
