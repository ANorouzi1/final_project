import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label
from tqdm.auto import tqdm

from cfgs import field_segmentation
from src.utils.postprocessing import instances_from_mask_and_sdf
from src.utils.utils import seed_everything


def _load_checkpoint(model, checkpoint, device):
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)


def _match_instances(pred, target, iou_threshold):
    pred_ids = [x for x in np.unique(pred) if x != 0]
    target_ids = [x for x in np.unique(target) if x != 0]
    pairs = []
    for pred_id in pred_ids:
        pred_mask = pred == pred_id
        for target_id in target_ids:
            target_mask = target == target_id
            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()
            iou = 0.0 if union == 0 else intersection / union
            if iou >= iou_threshold:
                pairs.append((iou, pred_id, target_id))
    pairs.sort(reverse=True)

    matched_pred = set()
    matched_target = set()
    iou_sum = 0.0
    for iou, pred_id, target_id in pairs:
        if pred_id in matched_pred or target_id in matched_target:
            continue
        matched_pred.add(pred_id)
        matched_target.add(target_id)
        iou_sum += iou

    tp = len(matched_pred)
    fp = len(pred_ids) - tp
    fn = len(target_ids) - tp
    f1_denom = 2 * tp + fp + fn
    pq_denom = tp + 0.5 * fp + 0.5 * fn
    return {
        "instance_f1": 1.0 if f1_denom == 0 else (2 * tp) / f1_denom,
        "pq": 1.0 if pq_denom == 0 else iou_sum / pq_denom,
        "pred_instances": len(pred_ids),
        "target_instances": len(target_ids),
    }


def _mask_boundary(mask, radius=2):
    kernel = 2 * radius + 1
    eroded = -torch.nn.functional.max_pool2d(-mask.float(), kernel_size=kernel, stride=1, padding=radius)
    return (mask.float() - eroded).clamp(min=0.0)


def _fast_mask_metric_sums(outputs, batch, threshold):
    pred = torch.sigmoid(outputs["mask_logits"]) > threshold
    target = batch["mask"] > 0.5
    intersection = (pred & target).flatten(1).sum(dim=1).float()
    union = (pred | target).flatten(1).sum(dim=1).float()
    denominator = pred.flatten(1).sum(dim=1).float() + target.flatten(1).sum(dim=1).float()
    pred_boundary = _mask_boundary(pred.float(), radius=2)
    target_boundary = _mask_boundary(target.float(), radius=2)
    boundary_intersection = (pred_boundary * target_boundary).flatten(1).sum(dim=1)
    boundary_union = ((pred_boundary + target_boundary) > 0).flatten(1).sum(dim=1).float()
    eps = 1e-6
    return {
        "miou": ((intersection + eps) / (union + eps)).sum().item(),
        "dice": ((2 * intersection + eps) / (denominator + eps)).sum().item(),
        "boundary_iou": ((boundary_intersection + eps) / (boundary_union + eps)).sum().item(),
    }


def _connected_component_scores(outputs, batch, threshold, iou_threshold):
    pred_batch = (torch.sigmoid(outputs["mask_logits"]) > threshold).detach().cpu().numpy()
    target_batch = batch["instance"].detach().cpu().numpy()
    scores = []
    for pred, target in zip(pred_batch, target_batch):
        pred_instances, _ = label(pred.squeeze() > 0)
        target_instances = target.squeeze()
        if target_instances.max() <= 1:
            target_instances, _ = label(target_instances > 0)
        scores.append(_match_instances(pred_instances, target_instances, iou_threshold))
    return scores


def _sdf_instance_scores(
    outputs,
    batch,
    mask_threshold,
    sdf_core_threshold,
    min_core_area,
    min_instance_area,
    iou_threshold,
):
    probs = torch.sigmoid(outputs["mask_logits"]).detach().cpu().numpy()
    sdf = torch.tanh(outputs["distance_logits"]).detach().cpu().numpy()
    targets = batch["instance"].detach().cpu().numpy()
    scores = []
    for prob, sdf_map, target in zip(probs, sdf, targets):
        pred_instances = instances_from_mask_and_sdf(
            prob.squeeze(),
            sdf_map.squeeze(),
            mask_threshold=mask_threshold,
            core_threshold=sdf_core_threshold,
            min_core_area=min_core_area,
            min_instance_area=min_instance_area,
        )
        target_instances = target.squeeze()
        if target_instances.max() <= 1:
            target_instances, _ = label(target_instances > 0)
        scores.append(_match_instances(pred_instances, target_instances, iou_threshold))
    return scores


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
    sdf_mask_threshold,
    sdf_core_threshold,
    min_core_area,
    min_instance_area,
    iou_threshold,
    include_instance,
):
    model.eval()
    metric_sums = {
        threshold: {"miou": 0.0, "dice": 0.0, "boundary_iou": 0.0, "instance_f1": 0.0, "pq": 0.0}
        for threshold in thresholds
    }
    sdf_sums = {}
    sdf_instance_sums = {}
    sdf_instance_counts = {}
    n_batches = 0
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
        n_batches += 1
        n_samples += batch_size

        for threshold in thresholds:
            for name, value in _fast_mask_metric_sums(outputs, batch, threshold).items():
                metric_sums[threshold][name] += value
            if include_instance:
                for score in _connected_component_scores(outputs, batch, threshold, iou_threshold):
                    metric_sums[threshold]["instance_f1"] += score["instance_f1"]
                    metric_sums[threshold]["pq"] += score["pq"]

        for key, value in _sdf_quality(outputs, batch).items():
            if not np.isnan(value):
                sdf_sums[key] = sdf_sums.get(key, 0.0) + value * batch_size

        if "distance_logits" in outputs:
            for threshold in thresholds:
                fusion_mask_threshold = threshold if sdf_mask_threshold is None else sdf_mask_threshold
                key_prefix = f"threshold_{threshold:.2f}"
                sdf_instance_sums.setdefault(
                    key_prefix,
                    {"instance_f1": 0.0, "pq": 0.0, "pred_instances": 0.0, "target_instances": 0.0},
                )
                sdf_instance_counts.setdefault(key_prefix, 0)
                for score in _sdf_instance_scores(
                    outputs,
                    batch,
                    fusion_mask_threshold,
                    sdf_core_threshold,
                    min_core_area,
                    min_instance_area,
                    iou_threshold,
                ):
                    for key in sdf_instance_sums[key_prefix]:
                        sdf_instance_sums[key_prefix][key] += score[key]
                    sdf_instance_counts[key_prefix] += 1

    mask_results = {
        threshold: {
            name: value / max(1, n_samples)
            for name, value in values.items()
            if include_instance or name not in ("instance_f1", "pq")
        }
        for threshold, values in metric_sums.items()
    }
    sdf_results = {
        key: value / max(1, n_samples)
        for key, value in sdf_sums.items()
    }
    sdf_instance_results = {}
    if sdf_instance_sums:
        sdf_instance_results = {
            threshold_key: {
                key: value / max(1, sdf_instance_counts[threshold_key])
                for key, value in values.items()
            }
            for threshold_key, values in sdf_instance_sums.items()
        }
    return mask_results, sdf_results, sdf_instance_results, n_samples


def _build_loader(config_name, batch_size):
    config = deepcopy(getattr(field_segmentation, config_name))
    config["data_args"]["shuffle"] = False
    config["data_args"]["num_workers"] = 0
    if batch_size is not None:
        config["data_args"]["batch_size"] = batch_size
    data_module = config["datamodule"](**config["data_args"])
    return config, data_module.get_heldout_loader()


def _print_mask_table(results, thresholds):
    has_instance = any(
        "instance_f1" in row
        for model_results in results.values()
        for row in model_results.values()
    )
    header = ["model", "threshold", "miou", "dice", "boundary_iou"]
    if has_instance:
        header.extend(["instance_f1", "pq"])
    print(" | ".join(header))
    print(" | ".join(["---"] * len(header)))
    for model_name, model_results in results.items():
        for threshold in thresholds:
            row = model_results[threshold]
            values = [
                model_name,
                f"{threshold:.2f}",
                f"{row['miou']:.4f}",
                f"{row['dice']:.4f}",
                f"{row['boundary_iou']:.4f}",
            ]
            if has_instance:
                values.extend([f"{row['instance_f1']:.4f}", f"{row['pq']:.4f}"])
            print(" | ".join(values))


def main():
    parser = argparse.ArgumentParser(description="Compare baseline and dual-head checkpoints on the same validation set.")
    parser.add_argument("--baseline-checkpoint", default="Saved/ftw_mask_baseline/best_model.pth")
    parser.add_argument("--dual-checkpoint", default="Saved/ftw_dual_head/best_model.pth")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--sdf-mask-threshold", type=float, default=None)
    parser.add_argument("--sdf-core-threshold", type=float, default=0.15)
    parser.add_argument("--sdf-min-core-area", type=int, default=4)
    parser.add_argument("--sdf-min-instance-area", type=int, default=20)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--include-instance", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    config, loader = _build_loader("ftw_dual_head", args.batch_size)

    models = {
        "baseline": ("ftw_mask_baseline", Path(args.baseline_checkpoint)),
        "dual_mask": ("ftw_dual_head", Path(args.dual_checkpoint)),
    }

    mask_results = {}
    sdf_results = {}
    sdf_instance_results = {}
    n_samples = 0
    for label_name, (config_name, checkpoint) in models.items():
        model_config = deepcopy(getattr(field_segmentation, config_name))
        model = model_config["model_arch"](**model_config["model_args"]).to(device)
        _load_checkpoint(model, checkpoint, device)
        print(f"Loaded {label_name}: {checkpoint}")
        results, sdf, sdf_instances, n_samples = evaluate_model(
            model,
            loader,
            device,
            args.thresholds,
            args.max_batches,
            args.sdf_mask_threshold,
            args.sdf_core_threshold,
            args.sdf_min_core_area,
            args.sdf_min_instance_area,
            args.iou_threshold,
            args.include_instance,
        )
        mask_results[label_name] = results
        if sdf:
            sdf_results[label_name] = sdf
        if sdf_instances:
            sdf_instance_results[label_name] = sdf_instances

    print(f"\nEvaluated samples: {n_samples}")
    print("\nMask-threshold metrics")
    _print_mask_table(mask_results, args.thresholds)

    if sdf_results:
        print("\nDual-head SDF quality")
        for model_name, values in sdf_results.items():
            print(model_name + " | " + " | ".join(f"{key}: {value:.4f}" for key, value in values.items()))

    if sdf_instance_results:
        print("\nDual-head SDF-fused instance metrics")
        for model_name, values in sdf_instance_results.items():
            support = "same as semantic threshold" if args.sdf_mask_threshold is None else f"{args.sdf_mask_threshold:.2f}"
            print(f"{model_name} | mask_support: {support} | core_threshold: {args.sdf_core_threshold:.2f}")
            for threshold_key, row in values.items():
                print("  " + threshold_key + " | " + " | ".join(f"{key}: {value:.4f}" for key, value in row.items()))


if __name__ == "__main__":
    main()
