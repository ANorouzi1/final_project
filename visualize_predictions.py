import argparse
import os
import tempfile
from copy import deepcopy
from pathlib import Path

import torch

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
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)


def _resolve_checkpoint(project_root, config, checkpoint):
    if checkpoint is not None:
        return Path(checkpoint)

    candidates = [
        project_root / "Saved" / config["name"] / "last_model.pth",
        project_root / "Saved" / "Saved" / config["name"] / "last_model.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


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
    parser.add_argument("--allow-random", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    project_root = Path(__file__).resolve().parent
    config = deepcopy(getattr(field_segmentation, args.config))
    config["data_args"]["shuffle"] = False
    config["data_args"]["num_workers"] = 0
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
