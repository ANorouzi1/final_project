# HLCV Project Speaker Notes

Approximate pacing: 45-60 seconds per main slide. The backup slide is only for questions.

## 1. Title

Introduce the project as agricultural field segmentation from Sentinel-2 imagery. The main theme is boundary quality: the output should preserve separate fields, not only predict field pixels.

## 2. What This Project Tries To Solve

Explain that the task is binary segmentation, but the useful map depends on boundaries. If neighboring fields merge, the prediction can have good-looking foreground overlap while still failing the parcel-level goal.

## 3. Why Boundaries Matter

Use the figure simply: the white mask tells us where fields are, while the colored map shows that those fields are separate parcels. The key point is that a prediction can cover the right field area but still be wrong if neighboring parcels merge.

## 4. Dataset And Supervision

Say that the input is the Sentinel-2 `window_a` image. The main target is the 2-class field/background mask, and the instance mask is used to build the signed distance field.

Fill later: final countries and chip counts: [.....]

## 5. Baseline

Emphasize that the mask-only U-Net is a fair reference. It uses BCE, Dice, and optional TV only; there is no SDF head and no SDF loss term.

## 6. Main Model

Describe the shared U-Net encoder/decoder and the two heads. The mask head predicts field pixels; the distance head predicts SDF geometry.

## 7. Loss

Explain that the main model uses SDF in the loss in two ways: the target SDF weights BCE near boundaries, and the SDF head is supervised with a SmoothL1 distance loss. Current main setting is boundary weight 20, sigma 0.12, and SDF SmoothL1 weight 0.1.

## 8. What I Changed

Summarize practical project work: FTW loader, SDF target/cache, mask-only and dual-head models, boundary loss, metrics, notebook workflow, and test/visualization scripts.

## 9. Experiment Grid

Present this as the ablation plan already encoded in the config file. It covers boundary weight, boundary width, SDF distance weight, augmentation, TV, and seam baseline controls.

Fill later: d003/d030/tv/no-aug run summaries: [.....]

## 10. Current Validation Results

State clearly that these are current validation results, not final test results. The boundary-aware run improves Boundary IoU from 0.3155 to 0.3591 and mIoU from 0.6672 to 0.6811.

Fill later: final test row: [.....]

## 11. Qualitative Results

Use one chip at a time. Green is correct, blue is false positive, and red is false negative. Compare the baseline columns with the boundary-aware prediction columns, focusing on boundary mistakes.

## 12. Interpretation

Be honest: the boundary-aware loss is clearly helpful in the current validation results, but the d0 ablation is close, so the final report should not overclaim that the SDF head alone caused the whole gain.

## 13. Final Results To Fill

Use this as a checklist before presenting. Replace all placeholders after final training and evaluation.

## 14. Conclusion

End with the story: field maps need boundaries, the method adds boundary supervision, validation supports the direction, and the final test result completes the evidence.

Final takeaway sentence: [.....]

## 15. Backup

Use only if asked about the broader France-only sweep. Make clear that France-only numbers should not be directly mixed with the multi-country validation numbers.
