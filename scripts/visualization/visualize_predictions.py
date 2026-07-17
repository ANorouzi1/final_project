import argparse
import os
import random
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import default_collate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "hlcv_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

from cfgs import field_segmentation
from src.utils.utils import seed_everything
from src.utils.visualization import show_predictions


def _load_checkpoint(model, checkpoint, device):
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)


def _resolve_checkpoint(project_root, config, checkpoint):
    if checkpoint is not None:
        return Path(checkpoint)

    candidate = project_root / "Saved" / config["name"] / "last_model.pth"
    return candidate if candidate.exists() else None


def _take_value(value, index):
    if torch.is_tensor(value):
        return value[index:index + 1]
    return [value[index]]


def _take_samples(loader, n_items, include_empty=False):
    samples = []
    for batch in loader:
        for index in range(batch["image"].shape[0]):
            if not include_empty and float(batch["mask"][index].sum()) == 0.0:
                continue
            samples.append({key: _take_value(value, index) for key, value in batch.items()})
            if len(samples) >= n_items:
                break
        if len(samples) >= n_items:
            break

    if not samples:
        raise ValueError("No non-empty samples were available to visualize.")

    merged = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples]
        merged[key] = torch.cat(values, dim=0) if torch.is_tensor(values[0]) else sum(values, [])
    return merged


def _base_dataset_and_index(dataset, index):
    while hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        index = dataset.indices[index]
        dataset = dataset.dataset
    return dataset, index


def _country_for_index(dataset, index):
    base, base_index = _base_dataset_and_index(dataset, index)
    if hasattr(base, "samples"):
        country, _ = base.samples[base_index]
        return country
    return "unknown"


def _random_index_order(dataset, seed, balanced_countries=False):
    rng = random.Random(seed)
    indices = list(range(len(dataset)))

    if not balanced_countries:
        rng.shuffle(indices)
        return indices

    by_country = {}
    for index in indices:
        by_country.setdefault(_country_for_index(dataset, index), []).append(index)
    for country_indices in by_country.values():
        rng.shuffle(country_indices)

    countries = sorted(by_country)
    rng.shuffle(countries)
    ordered = []
    while countries:
        next_countries = []
        for country in countries:
            ordered.append(by_country[country].pop())
            if by_country[country]:
                next_countries.append(country)
        countries = next_countries
    return ordered


def _take_random_samples(dataset, n_items, seed=0, include_empty=False, balanced_countries=False):
    samples = []
    for index in _random_index_order(dataset, seed, balanced_countries=balanced_countries):
        sample = dataset[index]
        if not include_empty and float(sample["mask"].sum()) == 0.0:
            continue
        samples.append(sample)
        if len(samples) >= n_items:
            break

    if not samples:
        raise ValueError("No non-empty samples were available to visualize.")
    return default_collate(samples)


def _split_values(values):
    if not values:
        return []
    out = []
    for value in values:
        out.extend(item.strip() for item in value.split(",") if item.strip())
    return out


def _take_samples_by_index(dataset, indices, include_empty=False):
    samples = []
    for index in indices:
        sample = dataset[index]
        if not include_empty and float(sample["mask"].sum()) == 0.0:
            raise ValueError(f"Sample index {index} is empty. Pass --include-empty to visualize it.")
        samples.append(sample)
    return default_collate(samples)


def _take_samples_by_id(dataset, sample_ids, include_empty=False):
    wanted = _split_values(sample_ids)
    index_by_id = {}
    index_by_aoi = {}
    duplicate_aois = set()

    for index in range(len(dataset)):
        base, base_index = _base_dataset_and_index(dataset, index)
        if not hasattr(base, "samples"):
            sample = dataset[index]
            index_by_id[sample["id"]] = index
            continue
        country, aoi_id = base.samples[base_index]
        index_by_id[f"{country}/{aoi_id}"] = index
        if aoi_id in index_by_aoi:
            duplicate_aois.add(aoi_id)
        else:
            index_by_aoi[aoi_id] = index

    indices = []
    for sample_id in wanted:
        if sample_id in index_by_id:
            indices.append(index_by_id[sample_id])
        elif sample_id in index_by_aoi and sample_id not in duplicate_aois:
            indices.append(index_by_aoi[sample_id])
        elif sample_id in duplicate_aois:
            raise ValueError(
                f"AOI id {sample_id!r} exists in multiple countries. Use the full country/aoi_id form."
            )
        else:
            raise ValueError(f"Sample id {sample_id!r} was not found in this split.")

    return _take_samples_by_index(dataset, indices, include_empty=include_empty)


def _select_loader(data_module, split):
    if split == "train":
        return data_module.get_loader()
    if split == "val":
        return data_module.get_heldout_loader()
    if split == "test":
        return data_module.get_test_loader()
    raise ValueError(f"Unknown split: {split}")


def main():
    parser = argparse.ArgumentParser(description="Save prediction-vs-ground-truth debug panels.")
    parser.add_argument("--config", default="ftw_dual_head")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--baseline-config", default="ftw_mask_baseline")
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=0)
    parser.add_argument("--mask-kind", choices=["semantic_2class", "semantic_3class"], default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--random-samples", action="store_true")
    parser.add_argument("--balanced-countries", action="store_true")
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--sample-index", type=int, action="append", default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    project_root = PROJECT_ROOT
    config = deepcopy(getattr(field_segmentation, args.config))
    config["data_args"]["shuffle"] = False
    config["data_args"]["num_workers"] = 0
    if "train_augment" in config["data_args"]:
        config["data_args"]["train_augment"] = False
    if args.mask_kind is not None:
        config["data_args"]["mask_kind"] = args.mask_kind

    data_module = config["datamodule"](**config["data_args"])
    loader = _select_loader(data_module, args.split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config["model_arch"](**config["model_args"]).to(device)

    checkpoint = _resolve_checkpoint(project_root, config, args.checkpoint)
    if checkpoint is not None:
        _load_checkpoint(model, checkpoint, device)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("Warning: no checkpoint found; visualizing random, untrained predictions.")

    baseline_model = None
    if not args.no_baseline:
        baseline_config = deepcopy(getattr(field_segmentation, args.baseline_config))
        baseline_model = baseline_config["model_arch"](**baseline_config["model_args"]).to(device)
        baseline_checkpoint = _resolve_checkpoint(project_root, baseline_config, args.baseline_checkpoint)
        if baseline_checkpoint is not None:
            _load_checkpoint(baseline_model, baseline_checkpoint, device)
            print(f"Loaded baseline checkpoint: {baseline_checkpoint}")
        else:
            baseline_model = None
            print("Warning: no baseline checkpoint found; skipping baseline columns.")

    if args.sample_id:
        batch = _take_samples_by_id(loader.dataset, args.sample_id, include_empty=args.include_empty)
    elif args.sample_index:
        batch = _take_samples_by_index(loader.dataset, args.sample_index, include_empty=args.include_empty)
    elif args.random_samples or args.balanced_countries:
        batch = _take_random_samples(
            loader.dataset,
            args.num_samples,
            seed=args.seed,
            include_empty=args.include_empty,
            balanced_countries=args.balanced_countries,
        )
    else:
        batch = _take_samples(loader, args.num_samples, include_empty=args.include_empty)
    print("Visualized samples:", ", ".join(batch.get("id", [])))
    print(f"Mask source: {config['data_args'].get('mask_kind', 'semantic_2class')}")
    fig = show_predictions(
        model,
        batch,
        device=device,
        threshold=args.threshold,
        max_items=args.num_samples,
        min_area=args.min_area,
        prediction_args={},
        baseline_model=baseline_model,
        baseline_label=baseline_config["name"] if baseline_model is not None else "baseline",
    )

    output_dir = Path(args.output_dir) if args.output_dir else project_root / "Visualizations" / config["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_predictions.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved visualization: {output_path}")


if __name__ == "__main__":
    main()
