from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

from .base_data_modules import BaseDataModule, subset_dataset


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


class FieldSegmentationDataset(Dataset):
    def __init__(self, data_dir, split="train", image_size=256, max_samples=None):
        self.root = Path(data_dir) / split
        self.image_size = image_size
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.distance_dir = self.root / "distances"

        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Expected images and masks under {self.root}. "
                "See README.md for the data layout."
            )

        self.image_paths = sorted(
            p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if max_samples:
            self.image_paths = self.image_paths[:max_samples]
        if not self.image_paths:
            raise FileNotFoundError(f"No image files found in {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self._matching_path(self.mask_dir, image_path.stem)
        distance_path = self._matching_path(self.distance_dir, image_path.stem, required=False)

        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize((self.image_size, self.image_size), Image.NEAREST)

        image_arr = np.asarray(image).astype(np.float32) / 255.0
        mask_arr = (np.asarray(mask).astype(np.float32) > 0).astype(np.float32)

        if distance_path:
            distance = Image.open(distance_path).convert("L").resize((self.image_size, self.image_size), Image.BILINEAR)
            distance_arr = np.asarray(distance).astype(np.float32) / 255.0
        else:
            distance_arr = normalized_distance_transform(mask_arr)

        image_tensor = torch.from_numpy(image_arr.transpose(2, 0, 1))
        mask_tensor = torch.from_numpy(mask_arr[None, ...])
        distance_tensor = torch.from_numpy(distance_arr[None, ...])

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "distance": distance_tensor,
            "id": image_path.stem,
        }

    @staticmethod
    def _matching_path(directory, stem, required=True):
        for suffix in IMAGE_EXTENSIONS:
            candidate = directory / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        if required:
            raise FileNotFoundError(f"Could not find mask for {stem} in {directory}")
        return None


def normalized_distance_transform(mask):
    distance = distance_transform_edt(mask > 0)
    max_value = float(distance.max())
    if max_value > 0:
        distance = distance / max_value
    return distance.astype(np.float32)


class FieldSegmentationDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir,
        split="train",
        image_size=256,
        heldout_split=0.0,
        max_samples=None,
        split_seed=42,
        **loader_kwargs,
    ):
        dataset = FieldSegmentationDataset(
            data_dir=data_dir,
            split=split,
            image_size=image_size,
            max_samples=max_samples,
        )
        dataset = subset_dataset(dataset, max_samples)
        super().__init__(dataset, heldout_split=heldout_split, split_seed=split_seed, **loader_kwargs)
