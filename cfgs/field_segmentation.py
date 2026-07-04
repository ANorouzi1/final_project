from copy import deepcopy
from functools import partial
from pathlib import Path

import torch

from src.data_loaders.field_dataset import FieldSegmentationDataModule, NUM_INPUT_CHANNELS
from src.data_loaders.synthetic_fields import SyntheticFieldDataModule
from src.losses.segmentation_losses import (
    BoundaryWeightedDiceBCEDistanceTVLoss,
    BoundaryWeightedSDFDiceBCEDistanceTVLoss,
    DiceBCETVLoss,
    DiceBCEDistanceTVLoss,
)
from src.metrics.segmentation_metrics import BoundaryIoU, MeanIoU
from src.models.unet import DualHeadUNet, FrameFieldUNet, MaskOnlyUNet
from src.trainers.field_trainer import FieldSegmentationTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDF_PREDICTION_ARGS = dict(use_sdf=True, sdf_weight=0.35, sdf_scale=4.0)


def _base_trainer(name, epochs, eval_period=5, monitor_metric="eval_boundary_iou"):
    return dict(
        n_gpu=1,
        epochs=epochs,
        eval_period=eval_period,
        save_dir=str(PROJECT_ROOT / "Saved"),
        save_checkpoints=False,
        save_best=True,
        monitor_metric=monitor_metric,
        monitor_mode="max",
        log_step=20,
        tensorboard=False,
        wandb=False,
    )


def _segmentation_metrics(prediction_args=None):
    prediction_args = prediction_args or {}
    return dict(
        miou=MeanIoU(threshold=0.5, **prediction_args),
        boundary_iou=BoundaryIoU(threshold=0.5, radius=2, **prediction_args),
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
    metrics=_segmentation_metrics(),
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
    metrics=_segmentation_metrics(),
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
        max_train_samples=8000,
        heldout_split=0.0,
        num_workers=6,
        sdf_cache_dir=str(PROJECT_ROOT / "sdf_cache"),
        train_augment=True,
        color_jitter=0.12,
    ),
    optimizer=partial(torch.optim.AdamW, lr=2e-4, weight_decay=1e-4),
    lr_scheduler=partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=40),
    criterion=DiceBCEDistanceTVLoss,
    criterion_args=dict(
        bce_weight=1.0,
        dice_weight=1.0,
        distance_weight=0.5,
        tv_weight=1e-6,
    ),
    metrics=_segmentation_metrics(),
    trainer_module=FieldSegmentationTrainer,
    trainer_config=_base_trainer("ftw_dual_head", epochs=50, eval_period=5),
)


ftw_dual_head_boundary_bce = deepcopy(ftw_dual_head)
ftw_dual_head_boundary_bce["name"] = "ftw_dual_head_boundary_bce"
ftw_dual_head_boundary_bce["criterion"] = BoundaryWeightedDiceBCEDistanceTVLoss
ftw_dual_head_boundary_bce["criterion_args"] = dict(
    bce_weight=1.0,
    dice_weight=1.0,
    distance_weight=0.5,
    tv_weight=1e-6,
    boundary_weight=3.0,
    boundary_sigma=0.12,
)
ftw_dual_head_boundary_bce["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce",
    epochs=50,
    eval_period=5,
)


ftw_dual_head_boundary_sdf = deepcopy(ftw_dual_head)
ftw_dual_head_boundary_sdf["name"] = "ftw_dual_head_boundary_sdf"
ftw_dual_head_boundary_sdf["criterion"] = BoundaryWeightedSDFDiceBCEDistanceTVLoss
ftw_dual_head_boundary_sdf["criterion_args"] = dict(
    bce_weight=1.0,
    dice_weight=1.0,
    distance_weight=0.5,
    tv_weight=1e-6,
    sdf_boundary_sigma=0.12,
)
ftw_dual_head_boundary_sdf["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_sdf",
    epochs=50,
    eval_period=5,
)


ftw_dual_head_sdf_prediction = deepcopy(ftw_dual_head)
ftw_dual_head_sdf_prediction["name"] = "ftw_dual_head_sdf_prediction"
ftw_dual_head_sdf_prediction["checkpoint_fallback_name"] = "ftw_dual_head"
ftw_dual_head_sdf_prediction["prediction_args"] = SDF_PREDICTION_ARGS
ftw_dual_head_sdf_prediction["metrics"] = _segmentation_metrics(
    prediction_args=SDF_PREDICTION_ARGS,
)
ftw_dual_head_sdf_prediction["trainer_config"] = _base_trainer(
    "ftw_dual_head_sdf_prediction",
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
        max_train_samples=8000,
        heldout_split=0.0,
        num_workers=6,
        sdf_cache_dir=str(PROJECT_ROOT / "sdf_cache"),
        train_augment=True,
        color_jitter=0.12,
    ),
    optimizer=partial(torch.optim.AdamW, lr=2e-4, weight_decay=1e-4),
    lr_scheduler=partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=40),
    criterion=DiceBCETVLoss,
    criterion_args=dict(
        bce_weight=1.0,
        dice_weight=1.0,
        tv_weight=1e-6,
    ),
    metrics=_segmentation_metrics(),
    trainer_module=FieldSegmentationTrainer,
    trainer_config=_base_trainer("ftw_mask_baseline", epochs=50, eval_period=5),
)
