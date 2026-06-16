from __future__ import annotations

from pathlib import Path

import numpy as np


def add_evaluate_args(parser):
    parser.add_argument("--images", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.set_defaults(func=run_evaluate)


def main(parser=None):
    if parser is not None:
        return add_evaluate_args(parser)
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate field segmentation checkpoint")
    add_evaluate_args(parser)
    return run_evaluate(parser.parse_args())


def run_evaluate(args) -> None:
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
    except Exception as exc:
        raise RuntimeError("Evaluation requires PyTorch. Install requirements.txt first.") from exc

    from .data import FieldFolderDataset, load_mask
    from .metrics import boundary_iou, instance_f1, mean_iou, panoptic_quality
    from .model import build_model
    from .postprocess import distance_seeded_instances

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(base_channels=checkpoint.get("base_channels", 32)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = FieldFolderDataset(args.images, args.masks, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    scores = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            outputs = model(image)
            seg_prob = torch.sigmoid(outputs["segmentation_logits"])
            seg_np = seg_prob[0, 0].cpu().numpy()
            dist_np = outputs["distance"][0, 0].cpu().numpy()
            true_mask = load_mask(batch["mask_path"][0])

            if true_mask.shape != seg_np.shape:
                seg_np = np.asarray(
                    F.interpolate(seg_prob, size=true_mask.shape[:2], mode="bilinear", align_corners=False)[0, 0].cpu()
                )
                dist_np = np.asarray(
                    F.interpolate(outputs["distance"], size=true_mask.shape[:2], mode="bilinear", align_corners=False)[0, 0].cpu()
                )

            pred_instances = distance_seeded_instances(seg_np, dist_np, mask_threshold=args.mask_threshold)
            scores.append(
                {
                    "miou": mean_iou(seg_np, true_mask, threshold=args.mask_threshold),
                    "boundary_iou": boundary_iou(seg_np, true_mask, threshold=args.mask_threshold),
                    "instance_f1": instance_f1(pred_instances, true_mask),
                    "pq": panoptic_quality(pred_instances, true_mask),
                }
            )

    means = {key: float(np.mean([s[key] for s in scores])) for key in scores[0]}
    for key, value in means.items():
        print(f"{key}: {value:.4f}")
