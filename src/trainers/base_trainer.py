import logging
import os
from abc import abstractmethod
from math import inf
from pathlib import Path

import torch

from src.utils.utils import prepare_device


class BaseTrainer:
    def __init__(self, config, log_dir):
        self.config = config
        self.trainer_config = config["trainer_config"]
        self.epochs = self.trainer_config["epochs"]
        self.save_period = self.trainer_config["save_period"]
        self.eval_period = self.trainer_config["eval_period"]
        self.history = []

        self._configure_logging(log_dir)
        self._configure_monitoring()

        self.checkpoint_dir = Path(self.trainer_config["save_dir"]) / self.config["name"]
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device, self.device_ids = prepare_device(self.trainer_config.get("n_gpu", 0))
        self.start_epoch = 1
        self.current_epoch = 1
        self.best_epoch = None

    def _configure_logging(self, log_dir):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.config["name"])
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        file_handler = logging.FileHandler(Path(log_dir) / f"{self.config['name']}.log")
        file_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))
        self.logger.addHandler(file_handler)

    def _configure_monitoring(self):
        self.monitor = self.trainer_config.get("monitor", "off")
        self.not_improved_count = 0
        self.early_stop = self.trainer_config.get("early_stop", inf)
        if self.early_stop <= 0:
            self.early_stop = inf
        if self.monitor == "off":
            self.monitor_mode = "off"
            self.monitor_metric = None
            self.monitor_best = None
        else:
            self.monitor_mode, self.monitor_metric = self.monitor.split()
            self.monitor_best = inf if self.monitor_mode == "min" else -inf

    def train(self):
        self.logger.info("------------ New Training Session ------------")
        for epoch in range(self.start_epoch, self.epochs + 1):
            self.current_epoch = epoch
            train_result = self._train_epoch()
            log = {"epoch": epoch, **train_result}

            if self.should_evaluate():
                eval_result = self.evaluate()
                log.update({f"eval_{key}": value for key, value in eval_result.items()})

            should_stop = self._process_monitor(log)
            self.history.append(log)
            self._log_epoch(log)

            if epoch % self.save_period == 0:
                self.save_model(self.checkpoint_dir / f"E{epoch}_model.pth")

            if should_stop:
                self.logger.info("Early stopping triggered.")
                break

        self.save_model(self.checkpoint_dir / "last_model.pth")
        return self.history

    def should_evaluate(self):
        return self.eval_period > 0 and self.current_epoch % self.eval_period == 0

    def _process_monitor(self, log):
        if self.monitor_mode == "off":
            return False
        if self.monitor_metric not in log:
            self.logger.warning("Monitor metric %s was not measured.", self.monitor_metric)
            return False

        current = log[self.monitor_metric]
        improved = (
            current < self.monitor_best if self.monitor_mode == "min" else current > self.monitor_best
        )
        if improved:
            self.monitor_best = current
            self.best_epoch = self.current_epoch
            self.not_improved_count = 0
            self.save_model(self.checkpoint_dir / "best_model.pth")
        else:
            self.not_improved_count += 1
        return self.not_improved_count > self.early_stop

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
        self.model.load_state_dict(state)

    @abstractmethod
    def _train_epoch(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, loader=None):
        raise NotImplementedError
