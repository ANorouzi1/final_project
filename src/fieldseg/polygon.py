from __future__ import annotations

import math
from collections.abc import Sequence

Point = tuple[float, float]


def _point_line_distance(point: Point, start: Point, end: Point) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    if dx == 0 and dy == 0:
        return math.hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / (dx * dx + dy * dy)))
    qx = sx + t * dx
    qy = sy + t * dy
    return math.hypot(px - qx, py - qy)


def douglas_peucker(points: Sequence[Point], epsilon: float) -> list[Point]:
    """Simplify a polyline with the Douglas-Peucker algorithm."""
    if len(points) <= 2:
        return list(points)

    start = points[0]
    end = points[-1]
    max_distance = -1.0
    split_index = 0

    for index in range(1, len(points) - 1):
        distance = _point_line_distance(points[index], start, end)
        if distance > max_distance:
            max_distance = distance
            split_index = index

    if max_distance > epsilon:
        left = douglas_peucker(points[: split_index + 1], epsilon)
        right = douglas_peucker(points[split_index:], epsilon)
        return left[:-1] + right
    return [start, end]


def close_polygon(points: Sequence[Point]) -> list[Point]:
    if not points:
        return []
    closed = list(points)
    if closed[0] != closed[-1]:
        closed.append(closed[0])
    return closed


def simplify_polygon(points: Sequence[Point], epsilon: float = 2.0) -> list[Point]:
    """Simplify a closed polygon while preserving closure."""
    closed = close_polygon(points)
    if len(closed) <= 4:
        return closed
    simplified = douglas_peucker(closed, epsilon)
    return close_polygon(simplified)
