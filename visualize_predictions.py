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


def main():
    parser = argparse.ArgumentParser(description="Save prediction-vs-ground-truth debug panels.")
    parser.add_argument("--config", default="ftw_dual_head")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-empty", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    project_root = Path(__file__).resolve().parent
    config = deepcopy(getattr(field_segmentation, args.config))
    config["data_args"]["shuffle"] = False
    config["data_args"]["num_workers"] = 0

    data_module = config["datamodule"](**config["data_args"])
    loader = data_module.get_heldout_loader() if args.split == "val" else data_module.get_loader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = config["model_arch"](**config["model_args"]).to(device)

    checkpoint = args.checkpoint
    if checkpoint is None:
        candidate = project_root / "Saved" / config["name"] / "best_model.pth"
        checkpoint = candidate if candidate.exists() else None
    else:
        checkpoint = Path(checkpoint)

    if checkpoint is not None:
        _load_checkpoint(model, checkpoint, device)
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("Warning: no checkpoint found; visualizing random, untrained predictions.")

    batch = _take_samples(loader, args.num_samples, include_empty=args.include_empty)
    fig = show_predictions(
        model,
        batch,
        device=device,
        threshold=args.threshold,
        max_items=args.num_samples,
    )

    output_dir = Path(args.output_dir) if args.output_dir else project_root / "Visualizations" / config["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_predictions.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved visualization: {output_path}")


if __name__ == "__main__":
    main()
