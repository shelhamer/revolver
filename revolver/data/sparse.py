import numpy as np

from torch.utils.data import Dataset

from .util import Wrapper


def pick_sparsity(target, count, ignore_index=255):
    """
    Pick |count| sparse points for every value in the target and bundle up as
    list of (value, indices) tuples.
    If count is a list of values, sample randomly from it
    """
    if isinstance(count, (list, tuple, np.ndarray)):
        count = np.random.choice(count)
    target = np.array(target, dtype=np.uint8)
    target_classes = np.unique(target[target != ignore_index])
    sparsity = []
    for cls in target_classes:
        cls_idx = np.where(target.flat == cls)[0]
        # skip masks that are already too sparse or tiny
        if len(cls_idx) < count:
            continue
        sparse_idxs = np.random.choice(cls_idx, size=count,
                                        replace=False)
        sparsity.append((cls, sparse_idxs))
    return sparsity

def sparsify(target, sparsity, ignore_index=255):
    """
    Reduce full target to the given sparse points and ignore everything else.
    """
    sparse_target = np.full_like(target, ignore_index)
    for cls, indices in sparsity:
        sparse_target.flat[indices] = cls
    return sparse_target


class SparseSeg(Wrapper, Dataset):
    """
    Sparsify targets for simulating low-data/few-shot settings.

    Sparsity is coded as (value, indices) tuples with target values and flat(!)
    indices into the original target.

    Args:
        dataset: the `Dataset` to sparsify
        count: the no. of points to *keep* for every value in the target
        if count is -1, return the original dense label
        static: whether to resample the sparse points on every load (False), or
                whether to pick fixed sparse points once-and-for-all (True)

    """

    def __init__(self, dataset, count=1, static=False):
        super().__init__(dataset)
        self.ds = dataset
        self.count = count
        self.static = static
        if static:
            # pick fixed sparse points for every element in the dataset
            self.sparsity = []
            for _, target, _ in self.ds:
                self.sparsity.append(pick_sparsity(target, self.count,
                                                   self.ds.ignore_index))

    def __getitem__(self, idx):
        im, target, aux = self.ds[idx]
        if self.count == -1:
            return im, target, aux
        if self.static:
            sparsity = self.sparsity[idx]
        else:
            sparsity = pick_sparsity(target, self.count)
        sparse_target = sparsify(target, sparsity, self.ds.ignore_index)
        return im, sparse_target, aux

    def __len__(self):
        return len(self.ds)
