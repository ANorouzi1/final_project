from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .distance import make_targets


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_image_files(path: str | Path) -> list[Path]:
    root = Path(path)
    return sorted(p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def load_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path))


class FieldFolderDataset:
    """PyTorch dataset for image/mask folders."""

    def __init__(self, image_dir: str | Path, mask_dir: str | Path, image_size: int | None = None):
        self.image_paths = list_image_files(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        try:
            import torch
            import torch.nn.functional as F
        except Exception as exc:
            raise RuntimeError("FieldFolderDataset requires PyTorch for training") from exc

        image_path = self.image_paths[index]
        mask_path = self.mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")

        image = load_image(image_path)
        mask = load_mask(mask_path)
        targets = make_targets(mask)

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()
        seg_t = torch.from_numpy(targets["segmentation"][None]).float()
        dist_t = torch.from_numpy(targets["distance"][None]).float()

        if self.image_size is not None:
            image_t = F.interpolate(image_t[None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
            seg_t = F.interpolate(seg_t[None], size=(self.image_size, self.image_size), mode="nearest")[0]
            dist_t = F.interpolate(dist_t[None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]

        return {
            "image": image_t,
            "segmentation": seg_t,
            "distance": dist_t,
            "mask_path": str(mask_path),
            "image_path": str(image_path),
        }
