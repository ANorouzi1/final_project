import logging
import os
from abc import abstractmethod
from pathlib import Path

import torch

from src.utils.utils import prepare_device


class BaseTrainer:
    def __init__(self, config, log_dir):
        self.config = config
        self.trainer_config = config["trainer_config"]
        self.epochs = self.trainer_config["epochs"]
        self.eval_period = self.trainer_config.get("eval_period", 0)
        self.save_checkpoints = self.trainer_config.get("save_checkpoints", False)
        self.checkpoint_period = max(
            1, int(self.trainer_config.get("checkpoint_period", 1))
        )
        self.save_best = self.trainer_config.get("save_best", True)
        self.monitor_metric = self.trainer_config.get("monitor_metric", "eval_boundary_iou")
        self.monitor_mode = self.trainer_config.get("monitor_mode", "max")
        self.best_metric = None
        self.best_epoch = None
        self.history = []

        self._configure_logging(log_dir)

        self.checkpoint_dir = Path(self.trainer_config["save_dir"]) / self.config["name"]
        if self.save_checkpoints or self.save_best:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device, self.device_ids = prepare_device(self.trainer_config.get("n_gpu", 0))
        self.start_epoch = 1
        self.current_epoch = 1

    def _configure_logging(self, log_dir):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.config["name"])
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        file_handler = logging.FileHandler(Path(log_dir) / f"{self.config['name']}.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))
        self.logger.addHandler(file_handler)

    def train(self):
        self.logger.info("------------ New Training Session ------------")
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.current_epoch = epoch
            train_result = self._train_epoch()
            log = {"epoch": epoch, **train_result}

            if self.should_evaluate():
                eval_result = self.evaluate()
                log.update({f"eval_{key}": value for key, value in eval_result.items()})

            self.history.append(log)
            self._update_best(log)
            self._log_epoch(log)
            if self.save_checkpoints and epoch % self.checkpoint_period == 0:
                self.save_checkpoint(self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth")
                self.save_checkpoint(self.checkpoint_dir / "checkpoint_last.pth")

        if self.save_checkpoints or self.save_best:
            self.save_model(self.checkpoint_dir / "last_model.pth")
        return self.history

    def _update_best(self, log):
        if not self.save_best or self.monitor_metric not in log:
            return

        current = log[self.monitor_metric]
        improved = (
            self.best_metric is None
            or (current > self.best_metric if self.monitor_mode == "max" else current < self.best_metric)
        )
        if not improved:
            return

        self.best_metric = current
        self.best_epoch = self.current_epoch
        self.save_model(self.checkpoint_dir / f"best_{self.monitor_metric}_model.pth")
        self.save_checkpoint(self.checkpoint_dir / f"best_{self.monitor_metric}_checkpoint.pth")

    def should_evaluate(self):
        return self.eval_period > 0 and self.current_epoch % self.eval_period == 0

    def _log_epoch(self, log):
        message = " | ".join(
            f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
            for key, value in log.items()
        )
        self.logger.info(message)
        print(message)

    def save_model(self, path):
        torch.save(self.model.state_dict(), os.fspath(path))

    def load_model(self, path):
        state = torch.load(os.fspath(path), map_location=self.device)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        if any(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        self.model.load_state_dict(state)

    def save_checkpoint(self, path):
        checkpoint = {
            "epoch": self.current_epoch,
            "history": self.history,
            "config_name": self.config.get("name"),
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if hasattr(self, "optimizer") else None,
            "lr_scheduler_state": self.lr_scheduler.state_dict() if hasattr(self, "lr_scheduler") else None,
        }
        torch.save(checkpoint, os.fspath(path))

    def load_checkpoint(self, path):
        checkpoint = torch.load(os.fspath(path), map_location=self.device)
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            self.load_model(path)
            return

        model_state = checkpoint["model_state"]
        if any(key.startswith("module.") for key in model_state):
            model_state = {key.removeprefix("module."): value for key, value in model_state.items()}
        self.model.load_state_dict(model_state)

        if checkpoint.get("optimizer_state") is not None and hasattr(self, "optimizer"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("lr_scheduler_state") is not None and hasattr(self, "lr_scheduler"):
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])

        self.history = checkpoint.get("history", [])
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.current_epoch = max(1, self.start_epoch)
        self.logger.info("Resumed from checkpoint %s at epoch %s", path, self.start_epoch)

    @abstractmethod
    def _train_epoch(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, loader=None):
        raise NotImplementedError
