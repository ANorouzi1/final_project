from copy import deepcopy
from functools import partial
from pathlib import Path

import torch

from src.data_loaders.field_dataset import FieldSegmentationDataModule, NUM_INPUT_CHANNELS
from src.data_loaders.synthetic_fields import SyntheticFieldDataModule
from src.losses.segmentation_losses import (
    BoundaryWeightedSDFDiceBCEDistanceTVLoss,
    DiceBCESeamLoss,
    DiceBCETVLoss,
    DiceBCEDistanceTVLoss,
)
from src.metrics.segmentation_metrics import BoundaryIoU, MeanIoU, PixelIoU
from src.models.unet import DualHeadUNet, FrameFieldUNet, MaskOnlyUNet
from src.trainers.field_trainer import FieldSegmentationTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FTW_DISTANCE_WEIGHT = 0.1
FTW_TV_WEIGHT = 0.0


def _base_trainer(name, epochs, eval_period=5):
    return dict(
        n_gpu=1,
        epochs=epochs,
        eval_period=eval_period,
        save_dir=str(PROJECT_ROOT / "Saved"),
        log_step=20,
        tensorboard=False,
        wandb=False,
    )


synthetic_debug = dict(
    name="synthetic_debug",
    model_arch=DualHeadUNet,
    model_args=dict(
        in_channels=3,
        base_channels=24,
        num_classes=1,
        bilinear=True,
    ),
    datamodule=SyntheticFieldDataModule,
    data_args=dict(
        n_samples=96,
        image_size=128,
        batch_size=8,
        shuffle=True,
        heldout_split=0.2,
        num_workers=0,
        seed=7,
    ),
    optimizer=partial(torch.optim.AdamW, lr=3e-4, weight_decay=1e-4),
    lr_scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=4, gamma=0.75),
    criterion=DiceBCEDistanceTVLoss,
    criterion_args=dict(
        bce_weight=1.0,
        dice_weight=1.0,
        distance_weight=0.35,
        tv_weight=1e-6,
    ),
    metrics=dict(
        pixel_iou=PixelIoU(threshold=0.5),
        miou=MeanIoU(threshold=0.5),
        boundary_iou=BoundaryIoU(threshold=0.5, radius=2),
    ),
    trainer_module=FieldSegmentationTrainer,
    trainer_config=_base_trainer("synthetic_debug", epochs=2, eval_period=1),
)


synthetic_full = dict(
    name="synthetic_full",
    model_arch=DualHeadUNet,
    model_args=dict(
        in_channels=3,
        base_channels=32,
        num_classes=1,
        bilinear=True,
    ),
    datamodule=SyntheticFieldDataModule,
    data_args=dict(
        n_samples=1200,
        image_size=160,
        batch_size=8,
        shuffle=True,
        heldout_split=0.2,
        num_workers=0,
        seed=11,
    ),
    optimizer=partial(torch.optim.AdamW, lr=3e-4, weight_decay=1e-4),
    lr_scheduler=partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=25),
    criterion=DiceBCEDistanceTVLoss,
    criterion_args=dict(
        bce_weight=1.0,
        dice_weight=1.0,
        distance_weight=0.4,
        tv_weight=1e-6,
    ),
    metrics=dict(
        pixel_iou=PixelIoU(threshold=0.5),
        miou=MeanIoU(threshold=0.5),
        boundary_iou=BoundaryIoU(threshold=0.5, radius=2),
    ),
    trainer_module=FieldSegmentationTrainer,
    trainer_config=_base_trainer("synthetic_full", epochs=25, eval_period=5),
)


ftw_dual_head = dict(
    name="ftw_dual_head",
    model_arch=DualHeadUNet,
    model_args=dict(
        in_channels=NUM_INPUT_CHANNELS,
        base_channels=32,
        num_classes=1,
        bilinear=False,
    ),
    datamodule=FieldSegmentationDataModule,
    data_args=dict(
        data_dir=str(PROJECT_ROOT / "data" / "ftw"),
        image_size=256,
        batch_size=16,
        shuffle=True,
        max_train_samples=1000000,
        heldout_split=0.0,
        num_workers=6,
        sdf_cache_dir=str(PROJECT_ROOT / "sdf_cache"),
        transform_preset="FTW_WithAugmentation",
    ),
    optimizer=partial(torch.optim.AdamW, lr=2e-4, weight_decay=1e-3),
    lr_scheduler=partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=40),
    criterion=DiceBCEDistanceTVLoss,
    criterion_args=dict(
        bce_weight=1.0,
        dice_weight=1.0,
        distance_weight=FTW_DISTANCE_WEIGHT,
        tv_weight=FTW_TV_WEIGHT,
    ),
    metrics=dict(
        pixel_iou=PixelIoU(threshold=0.5),
        miou=MeanIoU(threshold=0.5),
        boundary_iou=BoundaryIoU(threshold=0.5, radius=2),
    ),
    trainer_module=FieldSegmentationTrainer,
    trainer_config=_base_trainer("ftw_dual_head", epochs=30, eval_period=5),
)


ftw_dual_head_boundary_sdf = deepcopy(ftw_dual_head)
ftw_dual_head_boundary_sdf["name"] = "ftw_dual_head_boundary_sdf"
ftw_dual_head_boundary_sdf["criterion"] = BoundaryWeightedSDFDiceBCEDistanceTVLoss
ftw_dual_head_boundary_sdf["criterion_args"] = dict(
    bce_weight=1.0,
    dice_weight=1.0,
    distance_weight=FTW_DISTANCE_WEIGHT,
    tv_weight=FTW_TV_WEIGHT,
    sdf_boundary_sigma=0.12,
)
ftw_dual_head_boundary_sdf["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_sdf",
    epochs=30,
    eval_period=5,
)


ftw_mask_baseline = dict(
    name="ftw_mask_baseline",
    model_arch=MaskOnlyUNet,
    model_args=dict(
        in_channels=NUM_INPUT_CHANNELS,
        base_channels=32,
        num_classes=1,
        bilinear=False,
    ),
    datamodule=FieldSegmentationDataModule,
    data_args=dict(
        data_dir=str(PROJECT_ROOT / "data" / "ftw"),
        image_size=256,
        batch_size=16,
        shuffle=True,
        max_train_samples=1000000,
        heldout_split=0.0,
        num_workers=6,
        sdf_cache_dir=str(PROJECT_ROOT / "sdf_cache"),
        transform_preset="FTW_WithAugmentation",
    ),
    optimizer=partial(torch.optim.AdamW, lr=2e-4, weight_decay=1e-3),
    lr_scheduler=partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=40),
    criterion=DiceBCETVLoss,
    criterion_args=dict(
        bce_weight=1.0,
        dice_weight=1.0,
        tv_weight=FTW_TV_WEIGHT,
    ),
    metrics=dict(
        pixel_iou=PixelIoU(threshold=0.5),
        miou=MeanIoU(threshold=0.5),
        boundary_iou=BoundaryIoU(threshold=0.5, radius=2),
    ),
    trainer_module=FieldSegmentationTrainer,
    trainer_config=_base_trainer("ftw_mask_baseline", epochs=30, eval_period=5),
)


# Seam-weighted variant of the mask baseline: same MaskOnlyUNet + BCE/Dice/TV,
# but the per-pixel BCE is multiplied by the U-Net seam weight map (Ronneberger
# 2015), so merging two touching fields becomes the most expensive mistake.
# See DiceBCESeamLoss and seam_weight_map. Note: under FTW_WithAugmentation the
# seam map is rebuilt each epoch (a warped instance map cannot reuse the cache);
# set transform_preset=None in data_args to make it cache once instead.
ftw_seam = deepcopy(ftw_mask_baseline)
ftw_seam["name"] = "ftw_seam"
ftw_seam["data_args"] = dict(ftw_seam["data_args"])
ftw_seam["data_args"]["with_seam_weight"] = True
ftw_seam["data_args"]["seam_cache_dir"] = str(PROJECT_ROOT / "seam_cache")
ftw_seam["data_args"]["seam_w0"] = 10.0
ftw_seam["data_args"]["seam_sigma"] = 5.0
ftw_seam["criterion"] = DiceBCESeamLoss
ftw_seam["criterion_args"] = dict(bce_weight=1.0, dice_weight=1.0, tv_weight=FTW_TV_WEIGHT)
ftw_seam["trainer_config"] = _base_trainer("ftw_seam", epochs=30, eval_period=5)
