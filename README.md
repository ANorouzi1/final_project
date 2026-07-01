# HLCV Project: TV-Regularized Agricultural Field Segmentation

This project implements agricultural field segmentation with a dual-head U-Net, signed-distance-field auxiliary learning, total-variation regularization, and compact region/boundary metrics.

The structure follows Assignment 3:

- `cfgs/`: experiment dictionaries
- `src/data_loaders/`: data modules and datasets
- `src/models/`: model definitions
- `src/losses/`: segmentation losses
- `src/metrics/`: mIoU and Boundary IoU
- `src/trainers/`: trainer with last-checkpoint saving
- `notebooks/`: notebook-first workflow
- `Logs/` and `Saved/`: training logs and checkpoints

## Recommended Workflow

Open these notebooks:

1. `notebooks/01_project_overview.ipynb`
2. `notebooks/02_train_synthetic_demo.ipynb`

The second notebook runs without downloading data by using a synthetic polygon-field dataset. This is useful for checking that the full training loop, SDF loss, TV regularization, compact metrics, visualization, and checkpointing work before plugging in the real Fields of The World data.

## Real Data Layout

For FTW-style data, place files like this:

```text
data/ftw/
  train/
    images/
      sample_001.png
    masks/
      sample_001.png
    distances/              # SDF targets are generated from instance masks
      sample_001.png
  val/
    images/
    masks/
    distances/
```

The FTW loader uses the semantic mask for foreground supervision and the instance mask to build a signed distance field in `[-1, 1]`. Positive values are field interiors, negative values are background, and values near zero mark field boundaries, including borders between touching instances.

## Quick Start

From inside `outputs/hlcv`, run the notebooks. If you prefer a short script:

```bash
python run_experiment.py --config synthetic_debug
```

The project is intentionally notebook-friendly, so the script is only a convenience wrapper.

## Feedback-Driven Diagnostics

Show the binary-mask merging/blob issue that motivates the auxiliary SDF head:

```bash
python visualize_problem_setup.py --config ftw_dual_head --split val --num-samples 3
```

Train the fair mask-only ablation baseline and the proposed dual-head model:

```bash
python run_experiment.py --config ftw_mask_baseline
python run_experiment.py --config ftw_dual_head
```

Compare their validation metrics after both runs:

```bash
python compare_experiments.py
```

See `docs/feedback_response.md` for the ablation protocol, post-processing fairness note, and related SDF/TV literature pointers.
