import numpy as np

from torch.utils.data import Dataset

from .util import Wrapper


class TargetMapper(Wrapper, Dataset):
    """
    Map target values according to exhaustive key/value mapping for
    translating, combining, or dropping targets.

    n.b. By default the target is passed through unchanged, so any key left out
    of the mappingso the mapping will be preserved in the mapped target.

    Args:
        dataset: the `Dataset` to map (which must be a `SegData`).
        mapping: dict with key/from: value/to pairs for mapping target
    """

    def __init__(self, dataset, mapping):
        super().__init__(dataset)
        self.ds = dataset
        self.mapping = mapping

    @property
    def num_classes(self):
        # note: only correct for complete mappings with a key for every class
        return len(set(self.mapping.values()))

    def __getitem__(self, idx):
        im, target, aux = self.ds[idx]
        # map target
        map_target = target.copy()
        for k, v in self.mapping.items():
            map_target[target == k] = v
        aux.update({'mapping': self.mapping, 'full_target': target})
        return im, map_target, aux

    def __len__(self):
        return len(self.ds)


class TargetFilter(Wrapper, Dataset):
    """
    Filter a dataset to only keep elements that contain certain targets by
    taking *every* element that has *any* of the values to keep.

    Args:
        dataset: the `Dataset` to filter (which must be a `SegData`).
        keep: list of values to keep
    """

    def __init__(self, dataset, keep):
        super().__init__(dataset)
        self.ds = dataset
        self.keep = set(keep)
        self.slugs = self.load_slugs()

    def load_slugs(self):
        slugs = []
        for i, data in enumerate(self.ds):
            target, aux = data[-2], data[-1]
            keep = False
            if 'cls' in aux:  # mask datasets
                keep = aux['cls'] in self.keep
            else:  # image datasets
                classes = set(np.unique(target))
                keep = len(self.keep & classes)
            if keep:
                slugs.append(i)
        return slugs

    def __getitem__(self, idx):
        inner_idx = self.slugs[idx]
        return self.ds[inner_idx]

    def __len__(self):
        return len(self.slugs)


class MultiFilter(Wrapper, Dataset):
    """
    Filter a dataset to only keep elements that contain targets
    containing multiple instance or class masks.

    n.b. do not use to filter a Mask dataset because they explicitly
    discard all but one instance/class.

    Args:
        dataset: the `Dataset` to filter (which must be a `SegData`).
    """

    def __init__(self, dataset):
        super().__init__(dataset)
        self.ds = dataset
        self.slugs = self.load_slugs()

    def load_slugs(self):
        slugs = []
        for i, data in enumerate(self.ds):
            target = data[-2]
            masks = np.unique(target)
            masks = masks[masks != 0]
            masks = masks[masks != self.ds.ignore_index]
            if len(masks) > 1:
                slugs.append(i)
        return slugs

    def __getitem__(self, idx):
        inner_idx = self.slugs[idx]
        return self.ds[inner_idx]

    def __len__(self):
        return len(self.slugs)


class SubSampler(Wrapper, Dataset):

    def __init__(self, dataset, num_sample):
        super().__init__(dataset)
        if num_sample > len(dataset):
            raise Exception('Number to sample is larger than the number of slugs')
        self.ds = dataset
        self.num_sample = num_sample
        self.slugs = self.load_slugs()

    def load_slugs(self):
        slugs = list(np.random.choice(len(self.ds), self.num_sample, replace=False))
        return slugs

    def __getitem__(self, idx):
        inner_idx = self.slugs[idx]
        return self.ds[inner_idx]

    def __len__(self):
        return len(self.slugs)

