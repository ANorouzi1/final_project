# HLCV final presentation — speaker notes

Target length: about 11:45–12:00. Backup slides are only for Q/A.

## 1. Title — 0:20

We study agricultural field segmentation from Sentinel-2 imagery. The central question is whether the boundaries between individual fields are learned better through an auxiliary signed-distance head or by directly increasing the segmentation loss near boundaries.

## 2. The challenge — 0:55

The ordinary task is field versus background segmentation, but the useful map must also preserve each individual field. In the figure, the middle mask tells us where agricultural land is. The right image shows the separate fields within that area. A prediction can therefore obtain reasonable region overlap and still be wrong by merging neighboring fields. Our goal is to improve border agreement without harming the field mask.

## 3. Data and supervision — 0:55

We use the Fields of The World benchmark and the three countries available in our local training setup: Austria, Croatia, and Denmark. Every chip contains red, green, blue, and near-infrared bands from two contrasting dates, giving eight input channels. The complete-pair loader gives 10,938 training, 1,347 validation, and 1,430 test chips. The semantic mask is the main target. FTW also provides an instance mask in which each individual field has a different pixel value. Our loader remaps those existing values to compact IDs and uses the result to create a normalized signed distance field: positive inside each field, zero near its boundary, and negative outside.

## 4. Related work and contribution — 0:55

We build on three ingredients: the FTW dataset, the U-Net encoder–decoder, and the general idea that distance-to-boundary information can complement region losses. Our contribution is the controlled implementation and comparison. We added the SDF target and regression head, implemented SDF-derived BCE weighting, and built the matched baselines, Boundary IoU metric, checkpoint evaluation, and visualizations. The project question is specifically where the boundary signal should enter training.

## 5. Three methods — 1:05

The first method is the mask-only baseline. I checked the actual trained configuration: its loss is ordinary BCE plus Dice. The second method adds an SDF regression head and a Smooth L1 term with weight 0.1. The third method returns to a single mask head but uses the SDF only to weight BCE near field boundaries. Data, augmentation, backbone width, optimizer, and evaluation code remain the same. This makes the comparison about the supervision strategy rather than model capacity or data.

## 6. Second SDF head — 1:15

The dual-head model shares the full U-Net feature extractor. One small head predicts mask logits; the other predicts signed distance. The mask receives BCE and Dice. The distance prediction is passed through tanh and compared with the target using Smooth L1. The hypothesis is multi-task learning: to predict the SDF, shared features should encode where each individual field ends. The head adds only 9,313 parameters, about 0.12 percent, and inference still uses the mask head. Therefore any mask improvement must come through the shared representation.

## 7. Boundary-weighted BCE — 1:20

The final method uses the same SDF target differently. Each pixel’s BCE is multiplied by a weight that decays exponentially with absolute signed distance. At the border the weight is 21. At one sigma, a distance of 0.12, it is still 8.36, and it approaches one far from the border. We divide by the sum of weights so the term stays normalized. Dice remains unweighted and preserves global overlap. The key difference is that boundary information now acts directly on the mask logits instead of reaching them indirectly through a regression task.

## 8. Evaluation protocol — 0:50

Validation is used only to choose the saved checkpoint. The held-out test split is then used for the reported comparison. All three models are evaluated on exactly 1,430 chips using the same threshold of 0.60 and a two-pixel boundary band. mIoU measures region overlap. Boundary IoU measures overlap between the predicted and target boundary bands. Using one threshold for all models avoids choosing a separate test operating point for each model.

## 9. Test results — 1:25

This is the main result. The mask baseline reaches 0.3313 Boundary IoU and 0.6572 mIoU. The ordinary two-head SDF model reaches 0.3254 Boundary IoU, so it is 0.59 percentage points below the baseline on this test checkpoint. Boundary-weighted BCE reaches 0.3755, an absolute improvement of 4.42 percentage points, while mIoU also increases by 1.03 points to 0.6675. Therefore the gain is not a trade where we improve only a thin boundary metric and damage the region mask.

## 10. What the second head contributed — 1:10

This slide resolves an important result mismatch. Earlier validation runs suggested a small improvement from SDF supervision, roughly the two-point gain we initially discussed, but that claim does not hold on the held-out test checkpoint. We also compare boundary-weighted BCE with and without the SDF head. The full dual-head version reaches 0.3739 Boundary IoU; removing the SDF head slightly improves it to 0.3755. Once the segmentation loss already emphasizes borders, the second head adds no measurable benefit. The evidence supports direct segmentation supervision, not auxiliary regression, as the source of the final gain.

## 11. Qualitative validation example — 0:55

This Austrian validation chip shows the original Sentinel-2 RGB image, the ground-truth field mask, the baseline segmentation, and the segmentation from our best boundary-weighted BCE model. The baseline merges or removes several thin separations between neighboring fields. The best model preserves more of those narrow gaps, although difficult borders remain. This image is only a qualitative illustration; the aggregate comparison on the previous slide is based on all 1,430 held-out test chips.

## 12. Conclusion and outlook — 0:50

The conclusion is simple: in this setup, direct boundary-weighted BCE together with Dice is more effective than an auxiliary signed-distance regression head. It improves Boundary IoU by 4.42 percentage points and mIoU by 1.03 points over the matched mask baseline. The selected final model is therefore the simpler mask-only boundary-BCE version. The strongest next checks are multiple random seeds, per-country results, and instance-separation post-processing.

## Backup slides

The threshold table shows that boundary-weighted BCE remains best at every tested threshold. The loss backup gives the general objective and the exact final weights. The reference slide documents the external work used in the project.
