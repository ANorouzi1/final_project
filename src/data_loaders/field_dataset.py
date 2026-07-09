import os
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
        sdf_cache_dir=None,
        augment=False,
        color_jitter=0.12,
        transform=None,
        with_seam_weight=False,
        seam_cache_dir=None,
        seam_w0=10.0,
        seam_sigma=5.0,
    ):
        self.root = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.normalize_scale = float(normalize_scale)
        self.mask_kind = mask_kind
        self.augment = augment
        self.color_jitter = float(color_jitter)
        self.transform = transform
        self.with_seam_weight = bool(with_seam_weight)
        self.seam_w0 = float(seam_w0)
        self.seam_sigma = float(seam_sigma)
        # Optional on-disk cache for the SDF target (compute once, reuse each epoch).
        self.sdf_cache_dir = Path(sdf_cache_dir) if sdf_cache_dir else None
        if self.sdf_cache_dir is not None:
            self.sdf_cache_dir.mkdir(parents=True, exist_ok=True)
        self.seam_cache_dir = Path(seam_cache_dir) if seam_cache_dir else None
        if self.seam_cache_dir is not None:
            self.seam_cache_dir.mkdir(parents=True, exist_ok=True)

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
        if self.augment:
            image = _color_jitter_rgb(image, strength=self.color_jitter)

        instance = self._read_instance(country, aoi_id)

        mask_arr = self._read_tif(
            country_dir / "label_masks" / self.mask_kind / f"{aoi_id}.tif"
        )[0]
        mask_arr = self._resize_hw(mask_arr, mode="nearest")
        binary_mask = (mask_arr > 0).astype(np.float32)

        distance = self._sdf(country, aoi_id, instance)

        sample = {
            "image": image,
            "mask": binary_mask[None, ...],
            "distance": distance[None, ...],
            "sdf": distance[None, ...],
            "instance": instance.astype(np.int64),
            "id": f"{country}/{aoi_id}",
        }
        if self.transform is not None:
            sample = self.transform(sample)
        if self.with_seam_weight:
            sample["seam_weight"] = self._seam_weight(country, aoi_id, sample["instance"])
        output = {
            "image": torch.as_tensor(sample["image"]),
            "mask": torch.as_tensor(sample["mask"]),
            "distance": torch.as_tensor(sample["distance"]),
            "sdf": torch.as_tensor(sample["sdf"]),
            "instance": torch.as_tensor(sample["instance"]).long(),
            "id": sample["id"],
        }
        if self.with_seam_weight:
            output["seam_weight"] = torch.as_tensor(sample["seam_weight"]).float()
        return output

    def _sdf(self, country, aoi_id, instance):
        """Signed distance field with an optional on-disk cache (compute once)."""
        if self.sdf_cache_dir is None:
            return signed_instance_distance_field(instance)
        path = self._sdf_cache_path(country, aoi_id)
        if path.exists():
            try:
                return np.load(path)
            except Exception:
                pass  # corrupt/partial write -> recompute
        dist = signed_instance_distance_field(instance)
        tmp = path.parent / (path.name + f".tmp{os.getpid()}")  # atomic write
        try:
            with open(tmp, "wb") as fh:
                np.save(fh, dist)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return dist

    def _sdf_cache_path(self, country, aoi_id):
        key = f"{country}_{aoi_id}_s{self.image_size}.npy".replace("/", "_")
        return self.sdf_cache_dir / key

    def _read_instance(self, country, aoi_id):
        country_dir = self.root / country
        instance = self._read_tif(
            country_dir / "label_masks" / "instance" / f"{aoi_id}.tif"
        )[0]
        instance = _relabel_instances(instance)
        return self._resize_hw(instance, mode="nearest").astype(np.int32)

    def precompute_sdf_cache(self, overwrite=False, max_items=None):
        """Populate the SDF cache without reading model input imagery."""
        if self.sdf_cache_dir is None:
            return {"computed": 0, "skipped": 0, "total": 0, "cache_dir": None}

        computed = 0
        skipped = 0
        samples = self.samples[:max_items] if max_items is not None else self.samples
        for country, aoi_id in samples:
            cache_path = self._sdf_cache_path(country, aoi_id)
            if cache_path.exists() and not overwrite:
                skipped += 1
                continue
            instance = self._read_instance(country, aoi_id)
            self._sdf(country, aoi_id, instance)
            computed += 1

        return {
            "computed": computed,
            "skipped": skipped,
            "total": len(samples),
            "cache_dir": str(self.sdf_cache_dir),
        }

    def _seam_weight(self, country, aoi_id, instance):
        """Instance-seam BCE weights, cached only when no transform is active."""
        if self.seam_cache_dir is None or self.transform is not None:
            return seam_weight_map(instance, w0=self.seam_w0, sigma=self.seam_sigma)[None, ...]
        path = self._seam_cache_path(country, aoi_id)
        if path.exists():
            try:
                return np.load(path)[None, ...]
            except Exception:
                pass
        weight = seam_weight_map(instance, w0=self.seam_w0, sigma=self.seam_sigma)
        tmp = path.parent / (path.name + f".tmp{os.getpid()}")
        try:
            with open(tmp, "wb") as fh:
                np.save(fh, weight)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return weight[None, ...]

    def _seam_cache_path(self, country, aoi_id):
        key = (
            f"{country}_{aoi_id}_s{self.image_size}"
            f"_w{self.seam_w0:g}_sig{self.seam_sigma:g}.npy"
        ).replace("/", "_")
        return self.seam_cache_dir / key

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


def seam_weight_map(instance_mask, w0=10.0, sigma=5.0):
    """U-Net-style seam weights from an instance mask.

    The map is 1 everywhere plus a Gaussian bump where two different instance
    regions are close. It follows the Ronneberger et al. idea of using the sum
    of distances to the two nearest objects.
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Building seam weights needs scipy. Install via `pip install -r requirements.txt`."
        ) from exc

    instance_mask = np.asarray(instance_mask).squeeze()
    instance_ids = [instance_id for instance_id in np.unique(instance_mask) if instance_id != 0]
    if len(instance_ids) < 2 or w0 <= 0:
        return np.ones(instance_mask.shape, dtype=np.float32)

    nearest = np.full(instance_mask.shape, np.inf, dtype=np.float32)
    second_nearest = np.full(instance_mask.shape, np.inf, dtype=np.float32)
    for instance_id in instance_ids:
        distance = distance_transform_edt(instance_mask != instance_id).astype(np.float32)
        closer = distance < nearest
        second_nearest = np.where(closer, nearest, np.minimum(second_nearest, distance))
        nearest = np.where(closer, distance, nearest)

    sigma = max(float(sigma), 1e-6)
    seam = float(w0) * np.exp(-((nearest + second_nearest) ** 2) / (2.0 * sigma ** 2))
    return (1.0 + seam).astype(np.float32)


def _color_jitter_rgb(image, strength=0.12):
    """Apply light brightness/contrast/color jitter to RGB bands only."""
    if strength <= 0:
        return image

    out = image.copy()
    rgb_indices = [idx for idx in (0, 1, 2, 4, 5, 6) if idx < out.shape[0]]
    if not rgb_indices:
        return out

    rgb = out[rgb_indices]
    brightness = np.random.uniform(1.0 - strength, 1.0 + strength)
    contrast = np.random.uniform(1.0 - strength, 1.0 + strength)
    channel_scale = np.random.uniform(1.0 - strength, 1.0 + strength, size=(len(rgb_indices), 1, 1))

    mean = rgb.mean(axis=(1, 2), keepdims=True)
    rgb = (rgb - mean) * contrast + mean
    rgb = rgb * brightness * channel_scale.astype(np.float32)
    out[rgb_indices] = np.clip(rgb, 0.0, 1.0)
    return out.astype(np.float32, copy=False)


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
        test_split="test",
        normalize_scale=DEFAULT_S2_SCALE,
        mask_kind="semantic_2class",
        max_samples=None,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None,
        heldout_split=0.0,
        split_seed=42,
        sdf_cache_dir=None,
        with_seam_weight=False,
        seam_cache_dir=None,
        seam_w0=10.0,
        seam_sigma=5.0,
        train_augment=False,
        color_jitter=0.12,
        transform_preset=None,
        train_transform=None,
        eval_transform=None,
        **loader_kwargs,
    ):
        if split is not None:
            train_split = split
        if max_train_samples is None:
            max_train_samples = max_samples
        train_transform, eval_transform = self._resolve_transforms(
            transform_preset,
            train_transform,
            eval_transform,
        )

        train_dataset = FTWFieldDataset(
            data_dir=data_dir,
            split=train_split,
            countries=countries,
            image_size=image_size,
            max_samples=max_train_samples,
            normalize_scale=normalize_scale,
            mask_kind=mask_kind,
            sdf_cache_dir=sdf_cache_dir,
            augment=train_augment,
            color_jitter=color_jitter,
            transform=train_transform,
            with_seam_weight=with_seam_weight,
            seam_cache_dir=seam_cache_dir,
            seam_w0=seam_w0,
            seam_sigma=seam_sigma,
        )
        self._sdf_datasets = [train_dataset]
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
                sdf_cache_dir=sdf_cache_dir,
                transform=eval_transform,
                with_seam_weight=with_seam_weight,
                seam_cache_dir=seam_cache_dir,
                seam_w0=seam_w0,
                seam_sigma=seam_sigma,
            )
            if len(val_dataset) > 0:
                self.heldout_set = val_dataset
                self._sdf_datasets.append(val_dataset)

        if test_split is not None:
            try:
                test_dataset = FTWFieldDataset(
                    data_dir=data_dir,
                    split=test_split,
                    countries=countries,
                    image_size=image_size,
                    max_samples=max_test_samples,
                    normalize_scale=normalize_scale,
                    mask_kind=mask_kind,
                    sdf_cache_dir=sdf_cache_dir,
                    transform=eval_transform,
                    with_seam_weight=with_seam_weight,
                    seam_cache_dir=seam_cache_dir,
                    seam_w0=seam_w0,
                    seam_sigma=seam_sigma,
                )
            except FileNotFoundError:
                test_dataset = None
            if test_dataset is not None and len(test_dataset) > 0:
                self.test_set = test_dataset
                self._test_sdf_dataset = test_dataset
            else:
                self._test_sdf_dataset = None
        else:
            self._test_sdf_dataset = None

    def _resolve_transforms(self, transform_preset, train_transform, eval_transform):
        if transform_preset is None:
            return train_transform, eval_transform
        from src.utils.segmentation_transform_presets import presets as transform_presets

        if transform_preset not in transform_presets:
            available = ", ".join(sorted(transform_presets))
            raise ValueError(
                f"Unknown transform_preset={transform_preset!r}. "
                f"Available presets: {available}"
            )
        preset = transform_presets[transform_preset]
        if train_transform is None:
            train_transform = preset.get("train")
        if eval_transform is None:
            eval_transform = preset.get("eval")
        return train_transform, eval_transform

    def precompute_sdf_cache(self, include_heldout=True, include_test=False, overwrite=False):
        """Precompute SDF targets for configured FTW datasets."""
        datasets = list(self._sdf_datasets if include_heldout else self._sdf_datasets[:1])
        if include_test and self._test_sdf_dataset is not None:
            datasets.append(self._test_sdf_dataset)

        total = {"computed": 0, "skipped": 0, "total": 0, "cache_dir": None}
        seen = set()
        for dataset in datasets:
            if id(dataset) in seen:
                continue
            seen.add(id(dataset))
            result = dataset.precompute_sdf_cache(overwrite=overwrite)
            total["computed"] += result["computed"]
            total["skipped"] += result["skipped"]
            total["total"] += result["total"]
            total["cache_dir"] = result["cache_dir"] or total["cache_dir"]
        return total


# Backwards-compatible aliases (the config and notebooks import these names).
FieldSegmentationDataset = FTWFieldDataset
FieldSegmentationDataModule = FTWFieldDataModule
