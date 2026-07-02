import math
import random

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .base_data_modules import BaseDataModule
from .distance_targets import signed_instance_distance_field


class SyntheticFieldDataset(Dataset):
    """Small polygon-field generator for notebook demos and smoke tests."""

    def __init__(self, n_samples=128, image_size=128, seed=0, min_fields=3, max_fields=8):
        self.n_samples = n_samples
        self.image_size = image_size
        self.seed = seed
        self.min_fields = min_fields
        self.max_fields = max_fields

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        rng = random.Random(self.seed + index)
        np_rng = np.random.default_rng(self.seed + index)
        size = self.image_size

        background = np.zeros((size, size, 3), dtype=np.float32)
        background[..., 0] = 0.20 + np_rng.normal(0, 0.015, (size, size))
        background[..., 1] = 0.34 + np_rng.normal(0, 0.020, (size, size))
        background[..., 2] = 0.18 + np_rng.normal(0, 0.015, (size, size))

        mask_image = Image.new("I", (size, size), 0)
        draw = ImageDraw.Draw(mask_image)
        image = background.copy()
        field_count = rng.randint(self.min_fields, self.max_fields)

        for field_id in range(1, field_count + 1):
            polygon = _random_polygon(rng, size)
            draw.polygon(polygon, fill=field_id)

            color = np.array([
                rng.uniform(0.18, 0.42),
                rng.uniform(0.45, 0.74),
                rng.uniform(0.16, 0.34),
            ], dtype=np.float32)
            poly_mask = Image.new("L", (size, size), 0)
            ImageDraw.Draw(poly_mask).polygon(polygon, fill=255)
            poly_mask_arr = np.asarray(poly_mask) > 0
            texture = np_rng.normal(0, 0.025, (size, size, 1)).astype(np.float32)
            image[poly_mask_arr] = color + texture[poly_mask_arr]

        instance_mask = np.asarray(mask_image, dtype=np.int32)
        binary_mask = (instance_mask > 0).astype(np.float32)
        distance = signed_instance_distance_field(instance_mask, outside_clip=size / 4)
        image = np.clip(image, 0.0, 1.0).astype(np.float32)

        return {
            "image": torch.from_numpy(image.transpose(2, 0, 1)),
            "mask": torch.from_numpy(binary_mask[None, ...]),
            "distance": torch.from_numpy(distance[None, ...]),
            "sdf": torch.from_numpy(distance[None, ...]),
            "instance": torch.from_numpy(instance_mask.astype(np.int64)),
            "id": f"synthetic_{index:04d}",
        }


def _random_polygon(rng, size):
    cx = rng.uniform(0.15 * size, 0.85 * size)
    cy = rng.uniform(0.15 * size, 0.85 * size)
    radius_x = rng.uniform(0.08 * size, 0.20 * size)
    radius_y = rng.uniform(0.06 * size, 0.18 * size)
    vertices = rng.randint(4, 7)
    angle_offset = rng.uniform(0, math.tau)
    points = []
    for i in range(vertices):
        angle = angle_offset + math.tau * i / vertices + rng.uniform(-0.18, 0.18)
        x = cx + math.cos(angle) * radius_x * rng.uniform(0.75, 1.2)
        y = cy + math.sin(angle) * radius_y * rng.uniform(0.75, 1.2)
        points.append((max(1, min(size - 2, x)), max(1, min(size - 2, y))))
    return points


class SyntheticFieldDataModule(BaseDataModule):
    def __init__(
        self,
        n_samples=128,
        image_size=128,
        heldout_split=0.2,
        test_samples=None,
        seed=0,
        split_seed=42,
        **loader_kwargs,
    ):
        dataset = SyntheticFieldDataset(n_samples=n_samples, image_size=image_size, seed=seed)
        super().__init__(dataset, heldout_split=heldout_split, split_seed=split_seed, **loader_kwargs)
        if test_samples is None:
            test_samples = max(1, int(round(n_samples * heldout_split)))
        self.test_set = SyntheticFieldDataset(
            n_samples=test_samples,
            image_size=image_size,
            seed=seed + 100000,
        )
