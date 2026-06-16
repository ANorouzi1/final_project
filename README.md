# HLCV Project: Polygon-Aware Agricultural Field Segmentation

This project implements the proposal in the CVPR-style PDF: agricultural field segmentation with a dual-head U-Net, distance-transform auxiliary learning, polygon-aware regularization, and region/boundary metrics.

The structure follows Assignment 3:

- `cfgs/`: experiment dictionaries
- `src/data_loaders/`: data modules and datasets
- `src/models/`: model definitions
- `src/losses/`: segmentation losses
- `src/metrics/`: mIoU, Dice, Boundary IoU, instance F1, PQ-style score
- `src/trainers/`: trainer with best/last checkpoint saving and early stopping
- `notebooks/`: notebook-first workflow
- `Logs/` and `Saved/`: training logs and checkpoints

## Recommended Workflow

Open these notebooks:

1. `notebooks/01_project_overview.ipynb`
2. `notebooks/02_train_synthetic_demo.ipynb`

The second notebook runs without downloading data by using a synthetic polygon-field dataset. This is useful for checking that the full training loop, loss, metrics, visualization, and checkpointing work before plugging in the real Fields of The World data.

## Real Data Layout

For FTW-style data, place files like this:

```text
data/ftw/
  train/
    images/
      sample_001.png
    masks/
      sample_001.png
    distances/              # optional; generated from masks when missing
      sample_001.png
  val/
    images/
    masks/
    distances/
```

Masks should be binary or instance masks. Any non-zero value is treated as field foreground. Distance maps are normalized to `[0, 1]`; if they are missing, the dataset computes them from the mask.

## Quick Start

From inside `outputs/hlcv`, run the notebooks. If you prefer a short script:

```bash
python run_experiment.py --config synthetic_debug
```

The project is intentionally notebook-friendly, so the script is only a convenience wrapper.
