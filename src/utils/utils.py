import os
import random

import numpy as np
import torch


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_device(n_gpu_use):
    if torch.backends.mps.is_available() and n_gpu_use > 0:
        return torch.device("mps"), [0]
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu > 0:
        n_gpu_use = min(n_gpu_use, n_gpu)
        return torch.device("cuda:0"), list(range(n_gpu_use))
    return torch.device("cpu"), []


class MetricTracker:
    def __init__(self, keys):
        self.keys = keys
        self.reset()

    def reset(self):
        self.data = {key: {"sum": 0.0, "count": 0} for key in self.keys}

    def update(self, key, value):
        if key not in self.data:
            self.data[key] = {"sum": 0.0, "count": 0}
        self.data[key]["sum"] += float(value)
        self.data[key]["count"] += 1

    def result(self):
        return {
            key: values["sum"] / max(1, values["count"])
            for key, values in self.data.items()
            if values["count"] > 0
        }
