from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _jittered_rectangle(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    jitter: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    return [
        (x0 + rng.randint(-jitter, jitter), y0 + rng.randint(-jitter, jitter)),
        (x1 + rng.randint(-jitter, jitter), y0 + rng.randint(-jitter, jitter)),
        (x1 + rng.randint(-jitter, jitter), y1 + rng.randint(-jitter, jitter)),
        (x0 + rng.randint(-jitter, jitter), y1 + rng.randint(-jitter, jitter)),
    ]


def make_synthetic_field(
    size: int = 128,
    rows: int = 3,
    cols: int = 3,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a toy satellite-like RGB image and instance mask."""
    rng = random.Random(seed)
    image = Image.new("RGB", (size, size), (90, 112, 82))
    mask = Image.new("I", (size, size), 0)
    draw_image = ImageDraw.Draw(image)
    draw_mask = ImageDraw.Draw(mask)

    margin = max(6, size // 16)
    cell_w = (size - 2 * margin) // cols
    cell_h = (size - 2 * margin) // rows
    instance_id = 1

    for row in range(rows):
        for col in range(cols):
            x0 = margin + col * cell_w + rng.randint(0, 3)
            y0 = margin + row * cell_h + rng.randint(0, 3)
            x1 = margin + (col + 1) * cell_w - rng.randint(1, 4)
            y1 = margin + (row + 1) * cell_h - rng.randint(1, 4)
            polygon = _jittered_rectangle(x0, y0, x1, y1, max(2, size // 40), rng)
            color = (
                rng.randint(92, 166),
                rng.randint(124, 186),
                rng.randint(68, 118),
            )
            draw_image.polygon(polygon, fill=color)
            draw_mask.polygon(polygon, fill=instance_id)
            instance_id += 1

    noise = np.random.default_rng(seed).normal(0, 7, (size, size, 3))
    image_arr = np.clip(np.asarray(image, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
    return image_arr, np.asarray(mask, dtype=np.int32)


def write_synthetic_dataset(root: str | Path, count: int = 16, size: int = 128) -> None:
    root = Path(root)
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for index in range(count):
        image, mask = make_synthetic_field(size=size, seed=index)
        Image.fromarray(image).save(image_dir / f"sample_{index:04d}.png")
        Image.fromarray(mask.astype(np.uint16)).save(mask_dir / f"sample_{index:04d}.png")
