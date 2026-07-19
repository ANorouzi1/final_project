# HLCV final presentation — speaker notes

Target length: about 11:45–12:00. Backup slides are only for Q/A.

## 1. Title — 0:20

We study agricultural field segmentation from Sentinel-2 imagery. Our question is where boundary information should enter training: through distance regression, by directly weighting the mask loss, or through an explicit boundary-prediction output.

## 2. The challenge — 0:50

The primary target is field versus background, but a useful map must preserve the thin gaps between neighboring fields. A model can obtain reasonable area overlap while merging separate fields into one region. Our goal is therefore to improve border agreement without damaging the overall field mask.

## 3. Data and supervision — 0:50

We use the Fields of The World data available for Austria, Croatia, and Denmark. Every 256-by-256 chip contains RGB and near-infrared bands from two seasonal images, giving eight input channels. We have 10,938 training, 1,347 validation, and 1,430 test chips. The binary field mask is the segmentation target. The dataset also provides an instance mask that distinguishes neighboring fields; we convert it into a normalized signed distance field, which lets us derive distances and a narrow boundary band.

## 4. Related work and contribution — 0:45

We build on FTW, the U-Net architecture, and prior distance-aware segmentation ideas. Our work is the controlled comparison in this repository. We implemented an SDF regression output, boundary-weighted BCE, an explicit boundary-prediction output, Boundary IoU evaluation, matched checkpoint testing, and qualitative comparisons. The focus is not a new backbone; it is how the loss exposes boundary information to the same backbone.

## 5. Four methods — 1:00

The baseline has one mask output and uses ordinary BCE plus Dice. The SDF method adds a distance-regression output and a Smooth L1 loss. Boundary BCE returns to one mask output, but weights each BCE pixel by its distance from a field border. The new model has two active outputs: a mask output and an explicit boundary output. Its mask still uses boundary-weighted BCE plus Dice, and the boundary output adds a class-balanced BCE term with weight two. Data, augmentation, U-Net width, optimizer, and evaluation code are held constant.

## 6. SDF regression output — 1:05

The first two-output model shares the full U-Net and then predicts both mask logits and signed distance. Tanh bounds the predicted distance to the same minus-one-to-one range as the normalized target. Smooth L1 is quadratic for small errors and linear for larger errors, so it is less sensitive to outliers than mean squared error. The hypothesis is multi-task learning: learning field geometry may improve the shared mask features. Segmentation at inference still comes from the mask output.

## 7. Boundary-weighted BCE — 1:10

Here the distance target is used to change the mask loss directly. A pixel exactly on a boundary receives BCE weight 21. At normalized distance 0.12 the weight is 8.36, and it approaches one far away. Dividing by the sum of weights keeps the loss normalized. Dice remains unweighted and preserves global region overlap. This is the central difference: boundary mistakes now directly dominate the gradients of the final mask output.

## 8. Explicit boundary output — 1:00

The new model has two active heads, mask and boundary. The boundary target is one whenever the absolute normalized distance is at most 0.12, and zero otherwise. Because this positive band is thin, its BCE is class-balanced. The complete objective is boundary-weighted BCE plus Dice for the mask, plus two times boundary BCE for the second output. At inference we still use the mask output for the final segmentation; the boundary task acts through the shared features during training.

## 9. Evaluation protocol — 0:45

Validation is used to choose each saved checkpoint. The reported comparison then uses the held-out test split of 1,430 chips. Every model uses the same probability threshold, 0.60, and a boundary evaluation tolerance of two pixels, which is approximately 20 metres at Sentinel-2 resolution. mIoU measures region overlap, while Boundary IoU measures overlap between predicted and target boundary bands. A common threshold avoids selecting a separate test operating point for each model.

## 10. Test results — 1:20

The baseline reaches 0.3313 Boundary IoU. The SDF regression model reaches 0.3254, which is 0.59 percentage points below the baseline. Boundary-weighted BCE raises Boundary IoU to 0.3755, a gain of 4.42 points, and gives the highest mIoU at 0.6675. The new mask-plus-boundary model reaches the best Boundary IoU, 0.3832. That is 5.19 points over baseline and 0.77 points over Boundary BCE alone. Its mIoU is 0.6634: still 0.62 points above baseline, but 0.41 points below Boundary BCE. So the explicit boundary output gives a modest additional boundary gain, with a small region-overlap tradeoff relative to the mask-only boundary model.

## 11. What the auxiliary task contributed — 0:55

The important distinction is the target of the second output. Regressing signed distance did not improve the selected test checkpoint. Directly weighting the mask gradients produced the large gain. Once that direct loss is present, explicitly predicting the boundary band adds another 0.77 points of Boundary IoU. Our evidence therefore supports a focused boundary-classification task, not a distance-regression task, when the evaluation goal is border agreement.

## 12. Qualitative comparison — 0:55

All four predictions here use the same Austria validation chip and threshold. The top row gives the image, target, and baseline. The bottom row shows the SDF model, Boundary BCE, and the new mask-plus-boundary model. The panels let us compare the same thin separations rather than choosing a different favorable image for each method. This is only a qualitative example; the numerical claims come from all 1,430 held-out test chips.

## 13. Conclusion and outlook — 0:50

The conclusion is that direct boundary supervision works better than auxiliary distance regression in this setup. Boundary-weighted BCE improves the baseline by 4.42 Boundary-IoU points, and the explicit boundary output reaches the best score at 0.3832, or 5.19 points above baseline. Boundary BCE alone still has the highest mIoU, so the final choice depends on whether boundary fidelity or region overlap is prioritized. Next we would repeat multiple seeds, report results by country, and study instance-separation post-processing.

## Backup slides

The threshold table shows that the new model has the best Boundary IoU at all four tested thresholds. The loss backup states the exact objectives of the two strongest models. The reference slide documents the external work used in the project.
