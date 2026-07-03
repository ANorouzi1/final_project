import argparse
import math
from copy import deepcopy
from pathlib import Path

import torch
from tqdm.auto import tqdm

from cfgs import field_segmentation
from src.utils.prediction import mask_probability_from_outputs
from src.utils.utils import seed_everything


def _load_checkpoint(model, checkpoint, device):
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)


def _mask_boundary(mask, radius=2):
    kernel = 2 * radius + 1
    eroded = -torch.nn.functional.max_pool2d(-mask.float(), kernel_size=kernel, stride=1, padding=radius)
    return (mask.float() - eroded).clamp(min=0.0)


def _fast_mask_metric_sums(outputs, batch, threshold, prediction_args=None):
    pred = mask_probability_from_outputs(outputs, **(prediction_args or {})) > threshold
    target = batch["mask"] > 0.5
    intersection = (pred & target).flatten(1).sum(dim=1).float()
    union = (pred | target).flatten(1).sum(dim=1).float()
    pred_boundary = _mask_boundary(pred.float(), radius=2)
    target_boundary = _mask_boundary(target.float(), radius=2)
    boundary_intersection = (pred_boundary * target_boundary).flatten(1).sum(dim=1)
    boundary_union = ((pred_boundary + target_boundary) > 0).flatten(1).sum(dim=1).float()
    eps = 1e-6
    return {
        "miou": ((intersection + eps) / (union + eps)).sum().item(),
        "boundary_iou": ((boundary_intersection + eps) / (boundary_union + eps)).sum().item(),
    }


def _sdf_quality(outputs, batch):
    if "distance_logits" not in outputs:
        return {}

    pred = torch.tanh(outputs["distance_logits"]).detach().cpu()
    target = batch["distance"].detach().cpu()
    foreground = batch["mask"].detach().cpu() > 0.5
    center = target > 0.6
    boundary = target.abs() < 0.12

    values = {
        "sdf_mae": torch.abs(pred - target).mean().item(),
        "sdf_fg_mae": torch.abs(pred[foreground] - target[foreground]).mean().item()
        if foreground.any() else float("nan"),
        "sdf_center_pred": pred[center].mean().item() if center.any() else float("nan"),
        "sdf_center_target": target[center].mean().item() if center.any() else float("nan"),
        "sdf_boundary_abs": pred[boundary].abs().mean().item() if boundary.any() else float("nan"),
    }
    return values


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    device,
    thresholds,
    max_batches,
    prediction_args=None,
):
    model.eval()
    metric_sums = {
        threshold: {"miou": 0.0, "boundary_iou": 0.0}
        for threshold in thresholds
    }
    sdf_sums = {}
    n_samples = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="eval", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        outputs = model(batch["image"])
        batch_size = batch["image"].shape[0]
        n_samples += batch_size

        for threshold in thresholds:
            for name, value in _fast_mask_metric_sums(outputs, batch, threshold, prediction_args).items():
                metric_sums[threshold][name] += value

        for key, value in _sdf_quality(outputs, batch).items():
            if not math.isnan(value):
                sdf_sums[key] = sdf_sums.get(key, 0.0) + value * batch_size

    mask_results = {
        threshold: {
            name: value / max(1, n_samples)
            for name, value in values.items()
        }
        for threshold, values in metric_sums.items()
    }
    sdf_results = {
        key: value / max(1, n_samples)
        for key, value in sdf_sums.items()
    }
    return mask_results, sdf_results, n_samples


def _build_loader(config_name, batch_size, split):
    config = deepcopy(getattr(field_segmentation, config_name))
    config["data_args"]["shuffle"] = False
    config["data_args"]["num_workers"] = 0
    if batch_size is not None:
        config["data_args"]["batch_size"] = batch_size
    data_module = config["datamodule"](**config["data_args"])
    if split == "val":
        loader = data_module.get_heldout_loader()
    elif split == "test":
        loader = data_module.get_test_loader()
    else:
        raise ValueError(f"Unknown split: {split}")
    return config, loader


def _print_mask_table(results, thresholds):
    header = ["model", "threshold", "miou", "boundary_iou"]
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))
    for model_name, model_results in results.items():
        for threshold in thresholds:
            row = model_results[threshold]
            values = [
                model_name,
                f"{threshold:.2f}",
                f"{row['miou']:.4f}",
                f"{row['boundary_iou']:.4f}",
            ]
            print(" | ".join(values))


def main():
    parser = argparse.ArgumentParser(description="Compare baseline and dual-head checkpoints on the same validation set.")
    parser.add_argument("--baseline-checkpoint", default="Saved/ftw_mask_baseline/last_model.pth")
    parser.add_argument("--dual-checkpoint", default="Saved/ftw_dual_head/last_model.pth")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    config, loader = _build_loader("ftw_dual_head", args.batch_size, args.split)

    models = {
        "baseline": ("ftw_mask_baseline", Path(args.baseline_checkpoint)),
        "dual_mask": ("ftw_dual_head", Path(args.dual_checkpoint)),
        "dual_sdf_pred": ("ftw_dual_head_sdf_prediction", Path(args.dual_checkpoint)),
    }

    mask_results = {}
    sdf_results = {}
    n_samples = 0
    for label_name, (config_name, checkpoint) in models.items():
        model_config = deepcopy(getattr(field_segmentation, config_name))
        model = model_config["model_arch"](**model_config["model_args"]).to(device)
        _load_checkpoint(model, checkpoint, device)
        print(f"Loaded {label_name}: {checkpoint}")
        results, sdf, n_samples = evaluate_model(
            model,
            loader,
            device,
            args.thresholds,
            args.max_batches,
            model_config.get("prediction_args", {}),
        )
        mask_results[label_name] = results
        if sdf:
            sdf_results[label_name] = sdf

    print(f"\nEvaluated split: {args.split}")
    print(f"Evaluated samples: {n_samples}")
    print("\nMask-threshold metrics")
    _print_mask_table(mask_results, args.thresholds)

    if sdf_results:
        print("\nDual-head SDF quality")
        for model_name, values in sdf_results.items():
            print(model_name + " | " + " | ".join(f"{key}: {value:.4f}" for key, value in values.items()))


if __name__ == "__main__":
    main()
