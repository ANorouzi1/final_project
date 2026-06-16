from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fieldseg.distance import distance_transform, make_targets
from fieldseg.metrics import boundary_iou, instance_f1, mean_iou, panoptic_quality
from fieldseg.postprocess import distance_seeded_instances, polygons_from_instances
from fieldseg.synthetic import make_synthetic_field


def main() -> None:
    image, mask = make_synthetic_field(size=96, seed=7)
    targets = make_targets(mask)
    dist = distance_transform(mask)
    instances = distance_seeded_instances(targets["segmentation"], dist, seed_threshold=0.25)
    polygons = polygons_from_instances(instances, epsilon=2.0)

    print(f"synthetic image: {image.shape}")
    print(f"field pixels: {int(targets['segmentation'].sum())}")
    print(f"distance range: {float(dist.min()):.3f}..{float(dist.max()):.3f}")
    print(f"instances: {int(instances.max())}")
    print(f"polygons: {len(polygons)}")
    print(f"mIoU self-check: {mean_iou(targets['segmentation'], mask):.3f}")
    print(f"Boundary IoU self-check: {boundary_iou(targets['segmentation'], mask):.3f}")
    print(f"Instance F1 self-check: {instance_f1(mask, mask):.3f}")
    print(f"Panoptic Quality self-check: {panoptic_quality(mask, mask):.3f}")

    try:
        import torch
        from fieldseg.model import build_model

        model = build_model(base_channels=8)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)[None]).float() / 255.0
        outputs = model(tensor)
        print(f"model segmentation output: {tuple(outputs['segmentation_logits'].shape)}")
        print(f"model distance output: {tuple(outputs['distance'].shape)}")
    except Exception as exc:
        print(f"model forward skipped: {exc}")


if __name__ == "__main__":
    main()
