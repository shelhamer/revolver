import numpy as np

from torch.utils.data import Dataset

from .util import Wrapper


class ClassBalance(Wrapper, Dataset):
    """
    Class-balanced sampling from the wrapped datasets by mapping a flat
    index to a tuple of (dataset, slug idx).
    Each dataset should sub-sample their slugs to have the same number
    as the dataset with the smallest number.
    """

    def __init__(self, datasets):
        super().__init__(datasets[0])
        if len(set([len(ds) for ds in datasets])) != 1:
            raise Exception('Cannot class-balance datasets of different lengths')
        self.ds_len = len(datasets[0])
        self.datasets = datasets

    def __getitem__(self, idx):
        return self.datasets[idx // self.ds_len][idx % self.ds_len]

    def __len__(self):
        return self.ds_len*len(self.datasets)
