import argparse
import os
import tempfile
from copy import deepcopy
from pathlib import Path

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "hlcv_matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np
import torch
import tifffile
from matplotlib.colors import ListedColormap
from scipy.ndimage import label

from cfgs import field_segmentation


def _display_image(image):
    image = image.detach().cpu().float()
    preview = image[0] if image.shape[0] == 1 else image[:3].permute(1, 2, 0)
    low = torch.quantile(preview, 0.02)
    high = torch.quantile(preview, 0.98)
    if float(high - low) > 1e-6:
        preview = (preview - low) / (high - low)
    return preview.clamp(0, 1)


def _colorize_labels(labels):
    labels = np.asarray(labels)
    max_label = int(labels.max())
    if max_label == 0:
        return labels, "gray"

    rng = np.random.default_rng(0)
    colors = np.zeros((max_label + 1, 4), dtype=np.float32)
    colors[0] = [0, 0, 0, 1]
    colors[1:] = rng.uniform(0.2, 1.0, size=(max_label, 4))
    colors[1:, 3] = 1.0
    return labels, ListedColormap(colors)


def _count_instances(instance):
    return int(np.count_nonzero(np.unique(instance)))


def _load_checkpoint(model, checkpoint, device):
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)


def _checkpoint_summary(path):
    stat = path.stat()
    modified = __import__("datetime").datetime.fromtimestamp(stat.st_mtime)
    return f"{path} (modified {modified:%Y-%m-%d %H:%M:%S}, {stat.st_size / 1_000_000:.1f} MB)"


def _base_dataset_and_index(dataset, index):
    while hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        index = dataset.indices[index]
        dataset = dataset.dataset
    return dataset, index


def _read_label_arrays(dataset, index):
    base, base_index = _base_dataset_and_index(dataset, index)
    if not hasattr(base, "samples"):
        sample = dataset[index]
        return sample["mask"][0].numpy() > 0.5, sample["instance"].numpy()

    country, aoi_id = base.samples[base_index]
    country_dir = base.root / country
    mask = tifffile.imread(country_dir / "label_masks" / base.mask_kind / f"{aoi_id}.tif")
    instance = tifffile.imread(country_dir / "label_masks" / "instance" / f"{aoi_id}.tif")
    return np.squeeze(mask) > 0, np.squeeze(instance)


def _find_blob_examples(dataset, n_items):
    candidates = []
    fallback_candidates = []
    for index in range(len(dataset)):
        target_mask, target_instances = _read_label_arrays(dataset, index)
        if not target_mask.any():
            continue
        n_instances = _count_instances(target_instances)
        binary_components, n_components = label(target_mask)
        merge_gap = n_instances - n_components
        fallback_candidates.append((n_instances, n_components, index, binary_components))
        if merge_gap <= 0:
            continue
        candidates.append((merge_gap, n_instances, n_components, index, binary_components))

    if candidates:
        candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
        selected = [
            (merge_gap, n_instances, n_components, index, binary_components, f"SDF target separates {merge_gap}")
            for merge_gap, n_instances, n_components, index, binary_components in candidates[:n_items]
        ]
    else:
        if not fallback_candidates:
            raise ValueError("No non-empty field examples found to visualize.")
        fallback_candidates.sort(reverse=True, key=lambda item: item[0])
        selected = [
            (0, n_instances, n_components, index, binary_components, "SDF zeroes field boundaries")
            for n_instances, n_components, index, binary_components in fallback_candidates[:n_items]
        ]

    examples = []
    for _, n_instances, n_components, index, binary_components, score_title in selected:
        sample = dataset[index]
        examples.append({
            "id": sample.get("id", f"sample_{index}"),
            "image": sample["image"],
            "mask": sample["mask"][0].numpy() > 0.5,
            "instance": sample["instance"].numpy(),
            "distance": sample["distance"][0].numpy(),
            "blob_labels": binary_components,
            "blob_title": f"{n_components} binary blobs",
            "score_title": score_title,
            "n_instances": n_instances,
            "pred_mask": None,
            "pred_labels": None,
        })
    return examples


def _prediction_merge_score(pred_components, target_instances):
    score = 0
    worst = 0
    for component_id in np.unique(pred_components):
        if component_id == 0:
            continue
        overlapped = np.unique(target_instances[pred_components == component_id])
        overlapped = overlapped[overlapped != 0]
        n_overlap = len(overlapped)
        worst = max(worst, n_overlap)
        if n_overlap > 1:
            score += n_overlap - 1
    return score, worst


def _find_prediction_blob_examples(model, loader, device, threshold, n_items):
    candidates = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            probs = torch.sigmoid(model(batch["image"].to(device))["mask_logits"]).cpu()
            pred_batch = probs[:, 0].numpy() > threshold
            for index in range(pred_batch.shape[0]):
                target_instances = batch["instance"][index].numpy()
                pred_components, n_pred_components = label(pred_batch[index])
                score, worst_overlap = _prediction_merge_score(pred_components, target_instances)
                if score <= 0:
                    continue
                candidates.append((
                    score,
                    worst_overlap,
                    {
                        "image": batch["image"][index],
                        "id": batch.get("id", [f"sample_{index}"])[index],
                        "mask": batch["mask"][index, 0].numpy() > 0.5,
                        "instance": target_instances,
                        "distance": batch["distance"][index, 0].numpy(),
                        "blob_labels": pred_components,
                        "blob_title": f"{n_pred_components} predicted blobs",
                        "score_title": f"worst blob covers {worst_overlap} fields",
                        "n_instances": _count_instances(target_instances),
                        "pred_mask": pred_batch[index],
                        "pred_labels": pred_components,
                    },
                ))
                if len(candidates) >= n_items:
                    break
            if len(candidates) >= n_items:
                break

    if not candidates:
        raise ValueError("No predicted blob examples found at this threshold.")

    candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return [item[2] for item in candidates[:n_items]]


def main():
    parser = argparse.ArgumentParser(description="Visualize the binary-mask blob/merging problem.")
    parser.add_argument("--config", default="ftw_dual_head")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-kind", choices=["semantic_2class", "semantic_3class"], default=None)
    parser.add_argument("--no-model", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config = deepcopy(getattr(field_segmentation, args.config))
    config["data_args"]["shuffle"] = False
    config["data_args"]["num_workers"] = 0
    if args.mask_kind is not None:
        config["data_args"]["mask_kind"] = args.mask_kind
    data_module = config["datamodule"](**config["data_args"])
    dataset = data_module.heldout_set if args.split == "val" else data_module.dataset

    checkpoint = None
    if not args.no_model:
        checkpoint = Path(args.checkpoint) if args.checkpoint else project_root / "Saved" / config["name"] / "last_model.pth"
        if not checkpoint.exists():
            checkpoint = None

    if checkpoint is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = config["model_arch"](**config["model_args"]).to(device)
        _load_checkpoint(model, checkpoint, device)
        loader = data_module.get_heldout_loader() if args.split == "val" else data_module.get_loader()
        examples = _find_prediction_blob_examples(model, loader, device, args.threshold, args.num_samples)
        print(f"Loaded checkpoint: {_checkpoint_summary(checkpoint)}")
    else:
        examples = _find_blob_examples(dataset, args.num_samples)
        print("No checkpoint loaded; visualizing ground-truth label layout only.")
    print(f"Mask source: {config['data_args'].get('mask_kind', 'semantic_2class')}")

    n_cols = 6 if examples[0]["pred_mask"] is not None else 5
    fig, axes = plt.subplots(len(examples), n_cols, figsize=(3.2 * n_cols, 3.4 * len(examples)))
    if len(examples) == 1:
        axes = axes[None, :]

    for row, example in enumerate(examples):
        image = _display_image(example["image"])
        target_mask = example["mask"]
        target_instances = example["instance"]
        distance = example["distance"]

        instance_labels, instance_cmap = _colorize_labels(target_instances)
        component_labels, component_cmap = _colorize_labels(example["blob_labels"])

        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"image\n{example['id']}")
        axes[row, 1].imshow(target_mask, cmap="gray")
        axes[row, 1].set_title("semantic mask")
        axes[row, 2].imshow(instance_labels, cmap=instance_cmap, interpolation="nearest")
        axes[row, 2].set_title(f"{example['n_instances']} field instances")
        axes[row, 3].imshow(component_labels, cmap=component_cmap, interpolation="nearest")
        axes[row, 3].set_title(example["blob_title"])
        axes[row, 4].imshow(distance, cmap="coolwarm", vmin=-1, vmax=1)
        axes[row, 4].set_title(example["score_title"])
        if example["pred_mask"] is not None:
            axes[row, 5].imshow(example["pred_mask"], cmap="gray")
            axes[row, 5].set_title("predicted mask blob")

        for ax in axes[row]:
            ax.axis("off")

    output_dir = Path(args.output_dir) if args.output_dir else project_root / "Visualizations" / "problem_setup"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}_blob_issue.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Saved problem setup visualization: {output_path}")


if __name__ == "__main__":
    main()
