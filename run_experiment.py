import argparse
from copy import deepcopy
from pathlib import Path

from cfgs import field_segmentation
from src.utils.utils import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="synthetic_debug")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    seed_everything(0)
    config = deepcopy(getattr(field_segmentation, args.config))
    data_module = config["datamodule"](**config["data_args"])
    train_loader = data_module.get_loader()
    eval_loader = data_module.get_heldout_loader()

    trainer = config["trainer_module"](
        config=config,
        log_dir=project_root / "Logs",
        train_loader=train_loader,
        eval_loader=eval_loader,
    )
    trainer.train()


if __name__ == "__main__":
    main()
