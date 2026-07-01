from functools import partial
from pathlib import Path

import torch

from src.data_loaders.field_dataset import FieldSegmentationDataModule, NUM_INPUT_CHANNELS
from src.data_loaders.synthetic_fields import SyntheticFieldDataModule
from src.losses.segmentation_losses import DiceBCETVLoss, DiceBCEDistanceTVLoss
from src.metrics.segmentation_metrics import BoundaryIoU, MeanIoU
from src.models.unet import DualHeadUNet, FrameFieldUNet, MaskOnlyUNet
from src.trainers.field_trainer import FieldSegmentationTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        tv_weight=0.02,
    ),
    metrics=dict(
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
        tv_weight=0.02,
    ),
    metrics=dict(
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
        max_train_samples=8000,
        heldout_split=0.1,
        num_workers=6,
    ),
    optimizer=partial(torch.optim.AdamW, lr=2e-4, weight_decay=1e-4),
    lr_scheduler=partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=40),
    criterion=DiceBCEDistanceTVLoss,
    criterion_args=dict(
        bce_weight=1.0,
        dice_weight=1.0,
        distance_weight=0.5,
        tv_weight=0.02,
    ),
    metrics=dict(
        miou=MeanIoU(threshold=0.5),
        boundary_iou=BoundaryIoU(threshold=0.5, radius=2),
    ),
    trainer_module=FieldSegmentationTrainer,
    trainer_config=_base_trainer("ftw_dual_head", epochs=30, eval_period=5),
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
        heldout_split=0.1,
        num_workers=6,
    ),
    optimizer=partial(torch.optim.AdamW, lr=2e-4, weight_decay=1e-4),
    lr_scheduler=partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=40),
    criterion=DiceBCETVLoss,
    criterion_args=dict(
        bce_weight=1.0,
        dice_weight=1.0,
        tv_weight=0.02,
    ),
    metrics=dict(
        miou=MeanIoU(threshold=0.5),
        boundary_iou=BoundaryIoU(threshold=0.5, radius=2),
    ),
    trainer_module=FieldSegmentationTrainer,
    trainer_config=_base_trainer("ftw_mask_baseline", epochs=30, eval_period=5),
)
