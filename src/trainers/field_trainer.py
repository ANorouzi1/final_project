import torch
from tqdm.auto import tqdm

from .base_trainer import BaseTrainer
from src.utils.utils import MetricTracker
from src.metrics.segmentation_metrics import InstanceF1, PanopticQualityApprox


class FieldSegmentationTrainer(BaseTrainer):
    def __init__(self, config, log_dir, train_loader, eval_loader=None):
        super().__init__(config, log_dir)

        self.model = config["model_arch"](**config["model_args"]).to(self.device)
        if self.device.type == "cuda" and len(self.device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)

        self.criterion = config["criterion"](**config["criterion_args"])
        self.optimizer = config["optimizer"](self.model.parameters())
        self.lr_scheduler = config["lr_scheduler"](self.optimizer)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.metric_functions = config["metrics"]
        self.log_step = self.trainer_config.get("log_step", 20)

        loss_keys = [
            "loss",
            "bce",
            "dice_loss",
            "distance_loss",
            "sdf_gradient_loss",
            "boundary_loss",
            "polygon_loss",
        ]
        self.loss_keys = loss_keys
        # Instance-level metrics (InstanceF1, PanopticQuality) do connected-component
        # labeling + instance matching on CPU -> very slow on the noisy predictions
        # in training, and meaningless there anyway. Compute them only on eval;
        # train logs just the cheap pixel metrics.
        self._expensive_metrics = (InstanceF1, PanopticQualityApprox)
        self.train_metric_functions = {
            name: fn for name, fn in self.metric_functions.items()
            if not isinstance(fn, self._expensive_metrics)
        }
        self.train_metrics = MetricTracker(loss_keys + list(self.train_metric_functions))
        self.eval_metrics = MetricTracker(loss_keys + list(self.train_metric_functions)) #if you want instanceF1/PQ just write self.metric_functions instead
        self.logger.info(self.model)

    def _train_epoch(self):
        self.model.train()
        self.train_metrics.reset()
        iterator = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}", leave=False)

        for batch in iterator:
            batch = self._move_batch(batch)
            outputs = self.model(batch["image"])
            loss_dict = self.criterion(outputs, batch)
            loss = loss_dict["loss"]

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            self._update_metrics(self.train_metrics, loss_dict, outputs, batch,
                                 self.train_metric_functions)
            iterator.set_postfix(loss=f"{loss.item():.4f}")

        self.lr_scheduler.step()
        return self.train_metrics.result()

    @torch.no_grad()
    def evaluate(self, loader=None):
        loader = loader or self.eval_loader
        if loader is None:
            raise ValueError("No evaluation loader was provided.")

        self.model.eval()
        self.eval_metrics.reset()

        for batch in tqdm(loader, desc="Eval", leave=False):
            batch = self._move_batch(batch)
            outputs = self.model(batch["image"])
            loss_dict = self.criterion(outputs, batch)
            self._update_metrics(self.eval_metrics, loss_dict, outputs, batch,
                                 self.metric_functions)

        return self.eval_metrics.result()

    def _move_batch(self, batch):
        moved = {}
        for key, value in batch.items():
            moved[key] = value.to(self.device) if torch.is_tensor(value) else value
        return moved

    def _update_metrics(self, tracker, loss_dict, outputs, batch, metric_functions):
        for key in self.loss_keys:
            if key in loss_dict:
                value = loss_dict[key]
                tracker.update(key, float(value.detach().cpu()))
        for metric_key, metric_fn in metric_functions.items():
            tracker.update(metric_key, metric_fn.compute(outputs, batch))
