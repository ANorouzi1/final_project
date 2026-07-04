# HLCV Project: TV-Regularized Agricultural Field Segmentation

This project implements agricultural field segmentation with a dual-head U-Net, signed-distance-field auxiliary learning, total-variation regularization, and compact region/boundary metrics.

The structure follows Assignment 3:

- `cfgs/`: experiment dictionaries
- `src/data_loaders/`: data modules and datasets
- `src/models/`: model definitions
- `src/losses/`: segmentation losses
- `src/metrics/`: mIoU and Boundary IoU
- `src/trainers/`: shared training loop
- `src/utils/`: prediction, post-processing, and plotting helpers
- `scripts/evaluation/`: checkpoint/log comparison scripts
- `scripts/visualization/`: project diagnostic visualizations
- `notebooks/`: notebook-first workflow
- `docs/`: proposal and course feedback
- `Logs/`: training logs
- `Visualizations/`: generated figures
- `Saved/`: optional/legacy checkpoints
- `sdf_cache/`: generated signed-distance-field targets

## Recommended Workflow

Open these notebooks:

1. `notebooks/01_project_overview.ipynb`
2. `notebooks/02_train_synthetic_demo.ipynb`

The second notebook runs without downloading data by using a synthetic polygon-field dataset. This is useful for checking that the full training loop, SDF loss, TV regularization, compact metrics, visualization, and checkpointing work before plugging in the real Fields of The World data.

## Real Data Layout

For Fields of The World data, place files like this:

```text
data/ftw/
  france/
    s2_images/
      window_a/
        <aoi_id>.tif
      window_b/
        <aoi_id>.tif
    label_masks/
      semantic_2class/
        <aoi_id>.tif
      instance/
        <aoi_id>.tif
```

The FTW loader uses the semantic mask for foreground supervision and the instance mask to build a signed distance field in `[-1, 1]`. Positive values are field interiors, negative values are background, and values near zero mark field boundaries, including borders between touching instances.

## Quick Start

From the project root, run the notebooks. If you prefer a short script:

```bash
.venv/bin/python run_experiment.py --config synthetic_debug
```

The project is intentionally notebook-friendly, so the script is only a convenience wrapper.

## Feedback-Driven Diagnostics

Show the binary-mask merging/blob issue that motivates the auxiliary SDF head:

```bash
.venv/bin/python scripts/visualization/visualize_problem_setup.py --config ftw_dual_head --split val --num-samples 3 --no-model
```

Train the fair mask-only ablation baseline and the proposed dual-head model:

```bash
.venv/bin/python run_experiment.py --config ftw_mask_baseline
.venv/bin/python run_experiment.py --config ftw_dual_head
.venv/bin/python run_experiment.py --config ftw_dual_head_boundary_bce
```

Compare validation metrics after the runs:

```bash
.venv/bin/python scripts/evaluation/compare_experiments.py
```

Evaluate saved checkpoints directly, if you decide to keep checkpoints for a run:

```bash
.venv/bin/python scripts/evaluation/evaluate_checkpoints.py --models baseline dual_mask --split test
```

Generate prediction panels:

```bash
.venv/bin/python scripts/visualization/visualize_predictions.py --config ftw_dual_head --split test --num-samples 6
```

See `docs/proposal.pdf` for the original project idea and `docs/feedback.txt` for the course feedback that motivated the current baseline/ablation work.
