from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .base_data_modules import BaseDataModule
from .distance_targets import signed_instance_distance_field


# Bands per Sentinel-2 GeoTIFF in FTW (Red, Green, Blue, NIR).
BANDS_PER_WINDOW = 4
# Two temporal windows are stacked, matching the torchgeo FTW loader.
NUM_INPUT_CHANNELS = 2 * BANDS_PER_WINDOW
# Rough scaling for Sentinel-2 surface reflectance into ~[0, 1].
DEFAULT_S2_SCALE = 3000.0


class FTWFieldDataset(Dataset):
    """Fields of The World (FTW) field-boundary dataset.

    Expects the official on-disk layout produced by ``ftw data download``::

        root/
          <country>/
            chips_<country>.parquet        # columns: aoi_id, split
            s2_images/
              window_a/<aoi_id>.tif        # 4-band Sentinel-2 (R, G, B, NIR)
              window_b/<aoi_id>.tif        # 4-band Sentinel-2 (R, G, B, NIR)
            label_masks/
              semantic_2class/<aoi_id>.tif # 0 = background, 1 = field
              semantic_3class/<aoi_id>.tif # 0 = bg, 1 = interior, 2 = boundary
              instance/<aoi_id>.tif        # per-field instance ids

    Following the dual-head design of this project:

    * ``image``    = concat(window_b, window_a) -> 8 channels (torchgeo order)
    * ``mask``     = binary field mask from ``semantic_2class``
    * ``distance`` = signed distance field from ``instance`` in [-1, 1]
    * ``instance`` = raw instance-id map (useful for visualization/post-processing)

    Train/val/test membership is read from the per-country parquet file. If the
    parquet is missing, every chip found under ``s2_images/window_a`` is used
    (the ``split`` argument is ignored in that fallback).
    """

    def __init__(
        self,
        data_dir,
        split="train",
        countries=None,
        image_size=256,
        max_samples=None,
        normalize_scale=DEFAULT_S2_SCALE,
        mask_kind="semantic_2class",
    ):
        self.root = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.normalize_scale = float(normalize_scale)
        self.mask_kind = mask_kind

        if not self.root.exists():
            raise FileNotFoundError(
                f"FTW data root {self.root} does not exist. Download it with "
                "`ftw data download` and point data_dir at the resulting `ftw/` folder. "
                "See README.md."
            )

        self.countries = self._discover_countries(countries)
        self.samples = self._collect_samples()
        if max_samples:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise FileNotFoundError(
                f"No FTW chips found for split='{split}' under {self.root} "
                f"(countries={self.countries})."
            )

    # -- index construction -------------------------------------------------

    def _discover_countries(self, countries):
        if countries:
            return list(countries)
        found = [p.name for p in sorted(self.root.iterdir())
                 if p.is_dir() and (p / "s2_images").exists()]
        if not found:
            raise FileNotFoundError(
                f"No country sub-folders with an s2_images/ directory under {self.root}."
            )
        return found

    def _collect_samples(self):
        samples = []
        for country in self.countries:
            country_dir = self.root / country
            window_a_dir = country_dir / "s2_images" / "window_a"
            window_b_dir = country_dir / "s2_images" / "window_b"
            instance_dir = country_dir / "label_masks" / "instance"
            mask_dir = country_dir / "label_masks" / self.mask_kind
            ids = self._split_ids(country_dir, country)
            # Both temporal windows and both label targets are required.
            # Some metadata rows reference chips that are incomplete on disk.
            for aoi_id in ids:
                if (
                    (window_a_dir / f"{aoi_id}.tif").exists()
                    and (window_b_dir / f"{aoi_id}.tif").exists()
                    and (instance_dir / f"{aoi_id}.tif").exists()
                    and (mask_dir / f"{aoi_id}.tif").exists()
                ):
                    samples.append((country, aoi_id))
        return samples

    def _split_ids(self, country_dir, country):
        parquet = country_dir / f"chips_{country}.parquet"
        if parquet.exists():
            try:
                import pandas as pd
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "Reading FTW split metadata needs pandas (+ pyarrow). "
                    "Install via `pip install -r requirements.txt`."
                ) from exc
            df = pd.read_parquet(parquet)
            if "split" in df.columns and self.split is not None:
                df = df[df["split"] == self.split]
            return [str(aoi_id) for aoi_id in df["aoi_id"].values]

        # Fallback: no parquet -> use every chip in window_a, ignore split.
        window_a_dir = country_dir / "s2_images" / "window_a"
        return sorted(p.stem for p in window_a_dir.glob("*.tif"))

    # -- sample loading -----------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        country, aoi_id = self.samples[index]
        country_dir = self.root / country

        win_a = self._read_tif(country_dir / "s2_images" / "window_a" / f"{aoi_id}.tif")
        win_b = self._read_tif(country_dir / "s2_images" / "window_b" / f"{aoi_id}.tif")
        # torchgeo concatenates window_b before window_a.
        image = np.concatenate([win_b, win_a], axis=0)
        image = image.astype(np.float32) / self.normalize_scale
        image = np.clip(image, 0.0, 1.0)
        image = self._resize_chw(image, mode="bilinear")

        instance = self._read_tif(
            country_dir / "label_masks" / "instance" / f"{aoi_id}.tif"
        )[0]
        # FTW stores instance ids as huge uint64 values; relabel to compact
        # 0..N ints (0 = background) so they survive int casts and resizing.
        instance = _relabel_instances(instance)
        instance = self._resize_hw(instance, mode="nearest").astype(np.int32)

        mask_arr = self._read_tif(
            country_dir / "label_masks" / self.mask_kind / f"{aoi_id}.tif"
        )[0]
        mask_arr = self._resize_hw(mask_arr, mode="nearest")
        binary_mask = (mask_arr > 0).astype(np.float32)

        distance = signed_instance_distance_field(instance)

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(binary_mask[None, ...]),
            "distance": torch.from_numpy(distance[None, ...]),
            "sdf": torch.from_numpy(distance[None, ...]),
            "instance": torch.from_numpy(instance.astype(np.int64)),
            "id": f"{country}/{aoi_id}",
        }

    def _read_tif(self, path):
        if not path.exists():
            raise FileNotFoundError(f"Missing FTW tile: {path}")
        try:
            import rasterio
        except ImportError:
            try:
                import tifffile
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "Reading FTW GeoTIFFs needs rasterio or tifffile. "
                    "Install via `pip install -r requirements.txt`."
                ) from exc
            arr = tifffile.imread(path)
            if arr.ndim == 2:
                return arr[None, ...]
            if arr.ndim == 3 and arr.shape[-1] <= 16:
                return np.moveaxis(arr, -1, 0)
            return arr
        else:
            with rasterio.open(path) as src:
                return src.read()  # (bands, H, W)

    def _resize_chw(self, arr, mode):
        if arr.shape[-2:] == (self.image_size, self.image_size):
            return arr
        tensor = torch.from_numpy(arr).unsqueeze(0).float()
        kwargs = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
        tensor = F.interpolate(tensor, size=(self.image_size, self.image_size), mode=mode, **kwargs)
        return tensor.squeeze(0).numpy()

    def _resize_hw(self, arr, mode):
        if arr.shape == (self.image_size, self.image_size):
            return arr
        tensor = torch.from_numpy(arr.astype(np.float32))[None, None]
        tensor = F.interpolate(tensor, size=(self.image_size, self.image_size), mode=mode)
        return tensor[0, 0].numpy()


def _relabel_instances(instance_mask):
    """Map arbitrary instance ids (FTW uses huge uint64) to compact 0..N ints.

    Background (0) is preserved; each distinct non-zero id becomes 1, 2, ... N.
    """
    out = np.zeros(instance_mask.shape, dtype=np.int32)
    next_id = 1
    for old_id in np.unique(instance_mask):
        if old_id == 0:
            continue
        out[instance_mask == old_id] = next_id
        next_id += 1
    return out


class FTWFieldDataModule(BaseDataModule):
    """Data module over the official FTW train/val splits.

    By default the official ``train`` split is the training set and the official
    ``val`` split is returned by ``get_heldout_loader``. Set ``heldout_split`` > 0
    to instead carve a random validation set out of the train split.
    """

    def __init__(
        self,
        data_dir,
        countries=None,
        image_size=256,
        split=None,
        train_split="train",
        val_split="val",
        normalize_scale=DEFAULT_S2_SCALE,
        mask_kind="semantic_2class",
        max_samples=None,
        max_train_samples=None,
        max_val_samples=None,
        heldout_split=0.0,
        split_seed=42,
        **loader_kwargs,
    ):
        if split is not None:
            train_split = split
        if max_train_samples is None:
            max_train_samples = max_samples

        train_dataset = FTWFieldDataset(
            data_dir=data_dir,
            split=train_split,
            countries=countries,
            image_size=image_size,
            max_samples=max_train_samples,
            normalize_scale=normalize_scale,
            mask_kind=mask_kind,
        )
        super().__init__(train_dataset, heldout_split=heldout_split, split_seed=split_seed, **loader_kwargs)

        # Prefer the official validation split unless a random heldout was requested.
        if not heldout_split and val_split is not None:
            val_dataset = FTWFieldDataset(
                data_dir=data_dir,
                split=val_split,
                countries=countries,
                image_size=image_size,
                max_samples=max_val_samples,
                normalize_scale=normalize_scale,
                mask_kind=mask_kind,
            )
            if len(val_dataset) > 0:
                self.heldout_set = val_dataset


# Backwards-compatible aliases (the config and notebooks import these names).
FieldSegmentationDataset = FTWFieldDataset
FieldSegmentationDataModule = FTWFieldDataModule
