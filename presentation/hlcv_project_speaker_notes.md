# HLCV final presentation — speaker notes

Target length: about 11:45–12:00. Backup slides are only for Q/A.

## 1. Title — 0:20

We study agricultural field segmentation from Sentinel-2 imagery. Our question is where boundary information should enter training: by directly weighting the mask loss, through distance regression, or through an explicit boundary-prediction output.

## 2. The challenge — 0:50

The primary target is field versus background, but a useful map must preserve the thin gaps between neighboring fields. A model can obtain reasonable area overlap while merging separate fields into one region. Our goal is therefore to improve border agreement without damaging the overall field mask.

## 3. Data and supervision — 0:50

We use the Fields of The World data available for Austria, Croatia, and Denmark. Every 256-by-256 chip contains RGB and near-infrared bands from two seasonal images, giving eight input channels. We have 10,938 training, 1,347 validation, and 1,430 test chips. The binary field mask is the segmentation target. The dataset also provides an instance mask that distinguishes neighboring fields; we convert it into a normalized signed distance field, which lets us derive distances and a narrow boundary band.

## 4. Related work and contribution — 0:45

We build on FTW, the U-Net architecture, and prior distance-aware segmentation ideas. Our contribution is a controlled comparison of three ways to introduce boundary information: direct boundary weighting of the mask BCE, boundary-focused signed-distance regression, and an explicit boundary-classification output. All three use the same backbone and are compared on the same held-out test split with Boundary IoU and mIoU. The focus is not a new backbone; it is where the boundary signal enters training.

## 5. Four methods — 1:00

The baseline has one mask output and uses ordinary BCE plus Dice. Boundary BCE keeps one mask output, but weights each BCE pixel by its distance from a field border. The SDF method instead adds a distance-regression output and a boundary-focused Smooth L1 loss with weight 0.5. The new model has two active outputs: a mask output and an explicit boundary output. Its mask still uses boundary-weighted BCE plus Dice, and the boundary output adds a class-balanced BCE term with weight two. Data, augmentation, U-Net width, optimizer, and evaluation code are held constant.

## 6. Boundary-weighted BCE — 1:10

Here the distance target is used to change the mask loss directly. A pixel exactly on a boundary receives BCE weight 21. At normalized distance 0.12 the weight is 8.36, and it approaches one far away. Dividing by the sum of weights keeps the loss normalized. Dice remains unweighted and preserves global region overlap. This is the central difference: boundary mistakes now directly dominate the gradients of the final mask output.

## 7. SDF regression output — 1:05

The second method adds another output to the shared U-Net and predicts both mask logits and signed distance. Tanh bounds the predicted distance to the same minus-one-to-one range as the normalized target. Smooth L1 is quadratic for small errors and linear for larger errors, so it is less sensitive to outliers than mean squared error. We weight each regression error by exponential distance from the border: the weight is one at the zero level set and decays away from it with scale 0.12. After normalizing by the sum of pixel weights, we multiply the weighted SDF loss by 0.5. The hypothesis is multi-task learning: learning field geometry near borders may improve the shared mask features. Segmentation at inference still comes from the mask output.

## 8. Explicit boundary output — 1:00

The new model has two active heads, mask and boundary. The boundary target is one whenever the absolute normalized distance is at most 0.12, and zero otherwise. Because this positive band is thin, its BCE is class-balanced. The complete objective is boundary-weighted BCE plus Dice for the mask, plus two times boundary BCE for the second output. At inference we still use the mask output for the final segmentation; the boundary task acts through the shared features during training.

## 9. Evaluation protocol — 0:45

Validation is used to choose each saved checkpoint. The reported comparison then uses the held-out test split of 1,430 chips. Every model uses the same probability threshold, 0.60. For Boundary IoU, we create a strict symmetric band extending one pixel inward and one pixel outward from each border, which is approximately 10 metres per side at Sentinel-2 resolution. mIoU measures region overlap, while Boundary IoU measures overlap between the predicted and target boundary bands. A common threshold avoids selecting a separate test operating point for each model.

## 10. Test results — 1:20

With the strict symmetric boundary band, the baseline reaches 0.3469 Boundary IoU. Boundary-weighted BCE raises it to 0.3926, a gain of 4.57 points, and gives the highest mIoU at 0.6675. The boundary-focused SDF regression model with loss weight 0.5 reaches 0.3501, a smaller gain of 0.32 points, while its mIoU falls by 0.73 points to 0.6499. The new mask-plus-boundary model reaches the best Boundary IoU, 0.4000. That is 5.31 points over baseline and 0.74 points over Boundary BCE alone. Its mIoU is 0.6634: still 0.62 points above baseline, but 0.41 points below Boundary BCE.

## 11. What the auxiliary task contributed — 0:55

The important distinction is where the boundary signal acts. Boundary-focused SDF regression with weight 0.5 gives a small 0.32-point Boundary-IoU gain, but costs 0.73 points of mIoU. Directly weighting the mask gradients produces a much larger gain. Once that direct loss is present, explicitly predicting the boundary band adds another 0.74 points of Boundary IoU. Our evidence therefore favors direct mask and boundary supervision over an indirect distance-regression task when the goal is border agreement.

## 12. Qualitative comparison — 0:55

All four predictions here use the same Austria validation chip and threshold. The top row gives the image, target, and baseline. The bottom row shows Boundary BCE, the SDF model with weight 0.5, and the new mask-plus-boundary model. The panels let us compare the same thin separations rather than choosing a different favorable image for each method. This is only a qualitative example; the numerical claims come from all 1,430 held-out test chips.

## 13. Conclusion and outlook — 0:50

The conclusion is that direct boundary supervision works better than auxiliary distance regression in this setup. Boundary-focused SDF regression with weight 0.5 gives only a small 0.32-point Boundary-IoU gain and reduces mIoU by 0.73 points. With the strict one-pixel symmetric evaluation band, Boundary-weighted BCE improves the baseline by 4.57 points, and the explicit boundary output reaches the best score at 0.4000, or 5.31 points above baseline. Boundary BCE alone still has the highest mIoU, so the final choice depends on whether boundary fidelity or region overlap is prioritized. Next we would tune the loss weights and other hyperparameters more thoroughly, train and evaluate on more data, report results by country, and study instance-separation post-processing.

## Backup slides

The threshold table reports the SDF model with weight 0.5 and shows that the new model has the best Boundary IoU at all four tested thresholds. The loss backup states the exact objectives of the two strongest models. The reference slide documents the external work used in the project.
