from copy import deepcopy

import torch
from torch.utils.data import DataLoader, Subset, random_split


class BaseDataModule:
    def __init__(self, dataset, heldout_split=0.0, split_seed=42, **loader_kwargs):
        self.dataset = dataset
        self.loader_kwargs = loader_kwargs
        self.n_samples = len(dataset)
        self.heldout_split = heldout_split
        self.heldout_set = None
        self.test_set = None

        if heldout_split:
            self.dataset, self.heldout_set = self._split_data(heldout_split, split_seed)

        self.heldout_kwargs = deepcopy(self.loader_kwargs)
        self.heldout_kwargs["shuffle"] = False
        self.test_kwargs = deepcopy(self.heldout_kwargs)

    def get_loader(self):
        return DataLoader(self.dataset, **self.loader_kwargs)

    def get_heldout_loader(self):
        if self.heldout_set is None:
            raise ValueError("No heldout split was configured.")
        return DataLoader(self.heldout_set, **self.heldout_kwargs)

    def get_test_loader(self):
        if self.test_set is None:
            raise ValueError("No test split was configured.")
        return DataLoader(self.test_set, **self.test_kwargs)

    def _split_data(self, split, seed):
        if isinstance(split, int):
            valid_len = split
        else:
            valid_len = int(round(self.n_samples * split))
        valid_len = max(1, min(valid_len, self.n_samples - 1))
        train_len = self.n_samples - valid_len
        generator = torch.Generator().manual_seed(seed)
        train_set, valid_set = random_split(self.dataset, [train_len, valid_len], generator=generator)
        self.n_samples = len(train_set)
        return train_set, valid_set


def subset_dataset(dataset, n_items):
    if n_items is None or n_items >= len(dataset):
        return dataset
    return Subset(dataset, list(range(n_items)))
