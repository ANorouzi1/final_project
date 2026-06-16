# Polygon-Aware Agricultural Field Segmentation

Implementation scaffold for the project proposal: a dual-head U-Net that predicts
both agricultural field masks and normalized distance-transform maps, followed by
instance separation and polygon-aware refinement.

## What is included

- Dual-head U-Net model with segmentation and distance-transform outputs.
- Distance-transform target generation for binary field masks.
- Combined segmentation, distance, and boundary-regularization losses.
- Instance post-processing from mask + distance predictions.
- Polygon simplification with the Douglas-Peucker algorithm.
- mIoU, Boundary IoU, Instance F1, and Panoptic Quality metrics.
- Synthetic polygon dataset generator for smoke testing.
- Training and evaluation command-line entry points.

## Quick smoke test

This repository includes a smoke test that uses only NumPy and Pillow:

```bash
python3 scripts/smoke_test.py
```

If PyTorch is installed, the same smoke test also performs a forward pass through
the dual-head U-Net.

## Install for training

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Expected dataset layout

The training code accepts simple image/mask folders. Masks should be binary or
instance masks; all non-zero pixels are treated as field pixels for segmentation,
and instance ids are used where available for metrics.

```text
data/
  train/
    images/
      sample_001.png
    masks/
      sample_001.png
  val/
    images/
      sample_101.png
    masks/
      sample_101.png
```

For FTW, export or symlink the European subset into this structure.

## Train

```bash
python3 main.py train \
  --train-images data/train/images \
  --train-masks data/train/masks \
  --val-images data/val/images \
  --val-masks data/val/masks \
  --epochs 30 \
  --batch-size 8 \
  --out outputs/checkpoints/dual_head_unet.pt
```

## Evaluate

```bash
python3 main.py evaluate \
  --images data/val/images \
  --masks data/val/masks \
  --checkpoint outputs/checkpoints/dual_head_unet.pt
```

## Project notes

The distance-transform head helps keep touching fields separable by teaching the
network where field interiors peak and where physical boundaries should fall to
zero. Post-processing uses these peaks as instance seeds, then simplifies each
instance contour into vector-ready polygon vertices.
