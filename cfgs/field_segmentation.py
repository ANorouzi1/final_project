from copy import deepcopy
from functools import partial
from pathlib import Path

import torch

from src.data_loaders.field_dataset import FieldSegmentationDataModule, NUM_INPUT_CHANNELS
from src.data_loaders.synthetic_fields import SyntheticFieldDataModule
from src.losses.segmentation_losses import (
    BoundaryWeightedDiceBCEDistanceTVLoss,
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


def _segmentation_metrics():
    return dict(
        pixel_iou=PixelIoU(threshold=0.5),
        miou=MeanIoU(threshold=0.5),
        boundary_iou=BoundaryIoU(threshold=0.5, radius=2),
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
    metrics=_segmentation_metrics(),
    trainer_module=FieldSegmentationTrainer,
    trainer_config=_base_trainer("ftw_dual_head", epochs=30, eval_period=5),
)

# currently maybe the best 
ftw_dual_head_boundary_bce = deepcopy(ftw_dual_head)
ftw_dual_head_boundary_bce["name"] = "ftw_dual_head_boundary_bce"
ftw_dual_head_boundary_bce["criterion"] = BoundaryWeightedDiceBCEDistanceTVLoss
ftw_dual_head_boundary_bce["criterion_args"] = dict(
    bce_weight=1.0,
    dice_weight=1.0,
    distance_weight=FTW_DISTANCE_WEIGHT,
    tv_weight=FTW_TV_WEIGHT,
    boundary_weight=3.0,
    boundary_sigma=0.12,
)
ftw_dual_head_boundary_bce["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce",
    epochs=30,
    eval_period=5,
)

# try different weights on the boundary loss to see if it helps
ftw_dual_head_boundary_bce_w1 = deepcopy(ftw_dual_head_boundary_bce)
ftw_dual_head_boundary_bce_w1["name"] = "ftw_dual_head_boundary_bce_w1"
ftw_dual_head_boundary_bce_w1["criterion_args"]["boundary_weight"] = 1.0
ftw_dual_head_boundary_bce_w1["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce_w1",
    epochs=30,
    eval_period=5,
)

ftw_dual_head_boundary_bce_w5 = deepcopy(ftw_dual_head_boundary_bce)
ftw_dual_head_boundary_bce_w5["name"] = "ftw_dual_head_boundary_bce_w5"
ftw_dual_head_boundary_bce_w5["criterion_args"]["boundary_weight"] = 5.0
ftw_dual_head_boundary_bce_w5["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce_w5",
    epochs=30,
    eval_period=5,
)

ftw_dual_head_boundary_bce_w10 = deepcopy(ftw_dual_head_boundary_bce)
ftw_dual_head_boundary_bce_w10["name"] = "ftw_dual_head_boundary_bce_w10"
ftw_dual_head_boundary_bce_w10["criterion_args"]["boundary_weight"] = 10.0
ftw_dual_head_boundary_bce_w10["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce_w10",
    epochs=30,
    eval_period=5,
)

ftw_dual_head_boundary_bce_w15 = deepcopy(ftw_dual_head_boundary_bce)
ftw_dual_head_boundary_bce_w15["name"] = "ftw_dual_head_boundary_bce_w15"
ftw_dual_head_boundary_bce_w15["criterion_args"]["boundary_weight"] = 15.0
ftw_dual_head_boundary_bce_w15["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce_w15",
    epochs=30,
    eval_period=5,
)

ftw_dual_head_boundary_bce_w18 = deepcopy(ftw_dual_head_boundary_bce)
ftw_dual_head_boundary_bce_w18["name"] = "ftw_dual_head_boundary_bce_w18"
ftw_dual_head_boundary_bce_w18["criterion_args"]["boundary_weight"] = 18.0
ftw_dual_head_boundary_bce_w18["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce_w18",
    epochs=30,
    eval_period=2,
)


ftw_dual_head_boundary_bce_w20 = deepcopy(ftw_dual_head_boundary_bce)
ftw_dual_head_boundary_bce_w20["name"] = "ftw_dual_head_boundary_bce_w20"
ftw_dual_head_boundary_bce_w20["criterion_args"]["boundary_weight"] = 20.0
ftw_dual_head_boundary_bce_w20["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce_w20",
    epochs=30,
    eval_period=2,
)

# Boundary-width sweep: keep boundary and SDF-distance weights fixed so only
# boundary_sigma changes between runs.
ftw_dual_head_boundary_bce_w20_s004 = deepcopy(ftw_dual_head_boundary_bce_w20)
ftw_dual_head_boundary_bce_w20_s004["name"] = "ftw_dual_head_boundary_bce_w20_s004"
ftw_dual_head_boundary_bce_w20_s004["criterion_args"]["distance_weight"] = 0.1
ftw_dual_head_boundary_bce_w20_s004["criterion_args"]["boundary_sigma"] = 0.04
ftw_dual_head_boundary_bce_w20_s004["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce_w20_s004", epochs=50, eval_period=2
)

ftw_dual_head_boundary_bce_w20_s006 = deepcopy(ftw_dual_head_boundary_bce_w20)
ftw_dual_head_boundary_bce_w20_s006["name"] = "ftw_dual_head_boundary_bce_w20_s006"
ftw_dual_head_boundary_bce_w20_s006["criterion_args"]["distance_weight"] = 0.1
ftw_dual_head_boundary_bce_w20_s006["criterion_args"]["boundary_sigma"] = 0.06
ftw_dual_head_boundary_bce_w20_s006["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce_w20_s006", epochs=50, eval_period=2
)

ftw_dual_head_boundary_bce_w20_s009 = deepcopy(ftw_dual_head_boundary_bce_w20)
ftw_dual_head_boundary_bce_w20_s009["name"] = "ftw_dual_head_boundary_bce_w20_s009"
ftw_dual_head_boundary_bce_w20_s009["criterion_args"]["distance_weight"] = 0.1
ftw_dual_head_boundary_bce_w20_s009["criterion_args"]["boundary_sigma"] = 0.09
ftw_dual_head_boundary_bce_w20_s009["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce_w20_s009", epochs=50, eval_period=2
)

ftw_dual_head_boundary_bce_w20_s012 = deepcopy(ftw_dual_head_boundary_bce_w20)
ftw_dual_head_boundary_bce_w20_s012["name"] = "ftw_dual_head_boundary_bce_w20_s012"
ftw_dual_head_boundary_bce_w20_s012["criterion_args"]["distance_weight"] = 0.1
ftw_dual_head_boundary_bce_w20_s012["criterion_args"]["boundary_sigma"] = 0.12
ftw_dual_head_boundary_bce_w20_s012["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce_w20_s012", epochs=50, eval_period=2
)

ftw_dual_head_boundary_bce_w20_s012_d0 = deepcopy(
    ftw_dual_head_boundary_bce_w20_s012
)
ftw_dual_head_boundary_bce_w20_s012_d0["name"] = (
    "ftw_dual_head_boundary_bce_w20_s012_d0"
)
ftw_dual_head_boundary_bce_w20_s012_d0["criterion_args"]["distance_weight"] = 0.0
ftw_dual_head_boundary_bce_w20_s012_d0["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce_w20_s012_d0", epochs=50, eval_period=2
)

ftw_dual_head_boundary_bce_w20_s018 = deepcopy(ftw_dual_head_boundary_bce_w20)
ftw_dual_head_boundary_bce_w20_s018["name"] = "ftw_dual_head_boundary_bce_w20_s018"
ftw_dual_head_boundary_bce_w20_s018["criterion_args"]["distance_weight"] = 0.1
ftw_dual_head_boundary_bce_w20_s018["criterion_args"]["boundary_sigma"] = 0.18
ftw_dual_head_boundary_bce_w20_s018["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce_w20_s018", epochs=50, eval_period=2
)

ftw_dual_head_boundary_bce_w20_s024 = deepcopy(ftw_dual_head_boundary_bce_w20)
ftw_dual_head_boundary_bce_w20_s024["name"] = "ftw_dual_head_boundary_bce_w20_s024"
ftw_dual_head_boundary_bce_w20_s024["criterion_args"]["distance_weight"] = 0.1
ftw_dual_head_boundary_bce_w20_s024["criterion_args"]["boundary_sigma"] = 0.24
ftw_dual_head_boundary_bce_w20_s024["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce_w20_s024", epochs=50, eval_period=2
)

ftw_dual_head_boundary_bce_w25 = deepcopy(ftw_dual_head_boundary_bce)
ftw_dual_head_boundary_bce_w25["name"] = "ftw_dual_head_boundary_bce_w25"
ftw_dual_head_boundary_bce_w25["criterion_args"]["boundary_weight"] = 25.0
ftw_dual_head_boundary_bce_w25["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce_w25",
    epochs=30,
    eval_period=2,
)

ftw_dual_head_boundary_bce_w25_d10 = deepcopy(ftw_dual_head_boundary_bce_w25)
ftw_dual_head_boundary_bce_w25_d10["name"] = "ftw_dual_head_boundary_bce_w25_d10"
ftw_dual_head_boundary_bce_w25_d10["criterion_args"]["distance_weight"] = 1.0
ftw_dual_head_boundary_bce_w25_d10["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce_w25_d10",
    epochs=30,
    eval_period=2,
)

ftw_dual_head_boundary_bce_w25_d50 = deepcopy(ftw_dual_head_boundary_bce_w25)
ftw_dual_head_boundary_bce_w25_d50["name"] = "ftw_dual_head_boundary_bce_w25_d50"
ftw_dual_head_boundary_bce_w25_d50["criterion_args"]["distance_weight"] = 5.0
ftw_dual_head_boundary_bce_w25_d50["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce_w25_d50",
    epochs=30,
    eval_period=2,
)

ftw_dual_head_boundary_bce_w20_d70 = deepcopy(ftw_dual_head_boundary_bce_w20)
ftw_dual_head_boundary_bce_w20_d70["name"] = "ftw_dual_head_boundary_bce_w20_d70"
ftw_dual_head_boundary_bce_w20_d70["criterion_args"]["distance_weight"] = 7.0
ftw_dual_head_boundary_bce_w20_d70["lr_scheduler"] = partial(
    torch.optim.lr_scheduler.CosineAnnealingLR,
    T_max=80,
)
ftw_dual_head_boundary_bce_w20_d70["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce_w20_d70",
    epochs=50,
    eval_period=2,
)
ftw_dual_head_boundary_bce_w20_d70["trainer_config"]["save_checkpoints"] = True



ftw_dual_head_boundary_bce_w20_d150 = deepcopy(ftw_dual_head_boundary_bce_w20)
ftw_dual_head_boundary_bce_w20_d150["name"] = "ftw_dual_head_boundary_bce_w20_d150"
ftw_dual_head_boundary_bce_w20_d150["criterion_args"]["distance_weight"] = 15.0
ftw_dual_head_boundary_bce_w20_d150["lr_scheduler"] = partial(
    torch.optim.lr_scheduler.CosineAnnealingLR,
    T_max=80,
)
ftw_dual_head_boundary_bce_w20_d150["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce_w20_d150",
    epochs=50,
    eval_period=2,
)
ftw_dual_head_boundary_bce_w20_d150["trainer_config"]["save_checkpoints"] = True
ftw_dual_head_boundary_bce_w20_d150["trainer_config"]["checkpoint_period"] = 5





ftw_dual_head_boundary_bce_w30 = deepcopy(ftw_dual_head_boundary_bce)
ftw_dual_head_boundary_bce_w30["name"] = "ftw_dual_head_boundary_bce_w30"
ftw_dual_head_boundary_bce_w30["criterion_args"]["boundary_weight"] = 30.0
ftw_dual_head_boundary_bce_w30["trainer_config"] = _base_trainer(
    "ftw_dual_head_boundary_bce_w30",
    epochs=30,
    eval_period=2,
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
    metrics=_segmentation_metrics(),
    trainer_module=FieldSegmentationTrainer,
    trainer_config=_base_trainer("ftw_mask_baseline", epochs=30, eval_period=5),
)


# Seam-weighted variant of the mask baseline: same MaskOnlyUNet + BCE/Dice/TV,
# but per-pixel BCE is multiplied by an instance-derived U-Net seam weight map.
ftw_seam = deepcopy(ftw_mask_baseline)
ftw_seam["name"] = "ftw_seam"
ftw_seam["data_args"] = dict(ftw_seam["data_args"])
ftw_seam["data_args"]["with_seam_weight"] = True
ftw_seam["data_args"]["seam_cache_dir"] = str(PROJECT_ROOT / "seam_cache")
ftw_seam["data_args"]["seam_w0"] = 10.0
ftw_seam["data_args"]["seam_sigma"] = 5.0
ftw_seam["criterion"] = DiceBCESeamLoss
ftw_seam["criterion_args"] = dict(
    bce_weight=1.0,
    dice_weight=1.0,
    tv_weight=FTW_TV_WEIGHT,
)
ftw_seam["trainer_config"] = _base_trainer("ftw_seam", epochs=30, eval_period=5)
