# Feedback Response Notes

## Problem Plot To Include

The project now has `visualize_problem_setup.py`, which generates examples of the exact failure mode:

```bash
python visualize_problem_setup.py --config ftw_dual_head --split val --num-samples 3
```

Output:

```text
Visualizations/problem_setup/val_blob_issue.png
```

Each row shows:

- the Sentinel-2 RGB preview,
- the binary semantic field mask,
- the true per-field instance labels,
- the predicted connected components when a checkpoint is available,
- the per-instance distance target,
- the predicted binary blob that is causing the merge.

The key point is that a binary foreground prediction can merge multiple real field instances into one blob. The auxiliary distance target gives the model a per-instance interior signal, so touching or nearby fields can still have separate centers and boundaries. If no checkpoint is available, the script falls back to data-only examples where binary connected components merge ground-truth instances.

## Required Ablation

If the contribution is the auxiliary distance head, the cleanest baseline is now:

```bash
python run_experiment.py --config ftw_mask_baseline
```

and the proposed model is:

```bash
python run_experiment.py --config ftw_dual_head
```

These use the same FTW subset, resolution, optimizer, scheduler, batch size, metrics, and training length. The only intended difference is:

- `ftw_mask_baseline`: U-Net mask head only, BCE + Dice + polygon smoothness.
- `ftw_dual_head`: same U-Net backbone, mask head plus distance head, BCE + Dice + foreground-only distance regression + polygon smoothness.

After both logs exist:

```bash
python compare_experiments.py
```

This prints the best validation row for mIoU, Dice, Boundary IoU, Instance F1, and PQ approximation.

## Shape-Representation Context

Useful directions for framing or extending the method:

- Signed distance fields represent shape boundaries implicitly as a zero-level set, with distance magnitude encoding how far a point is from the boundary. DeepSDF learns continuous SDFs for 3D shape classes, but the same idea motivates a 2D field-distance auxiliary target.
  Source: https://arxiv.org/abs/1901.05103

- Deep Watershed Transform trains a CNN to output an energy landscape where object instances become separable basins. This is close in spirit to using a learned distance/energy map to prevent merged blobs.
  Source: https://arxiv.org/abs/1611.08303

- StarDist predicts star-convex polygons from pixels and was designed to avoid merging crowded objects. A 2D agricultural-field variant could predict radial distances, polygon vertices, or field centers as an alternative to the current distance head.
  Source: https://arxiv.org/abs/1806.03535

- Polygon-RNN and PolyTransform are examples of explicit polygon prediction/refinement. They support the idea that polygon structure can be modeled directly instead of only through raster masks.
  Sources: https://arxiv.org/abs/1704.05548 and https://arxiv.org/abs/1912.02801

## Post-Processing Fairness

If watershed or another post-hoc splitting algorithm is added later, report it as a separate factor:

- baseline raw,
- dual-head raw,
- baseline + same post-processing,
- dual-head + same post-processing.

That keeps the contribution of the auxiliary head separate from the contribution of post-processing.
