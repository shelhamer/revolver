import numpy as np

from torch.utils.data import Dataset

from .seg import MaskSemSeg
from .filter import TargetFilter
from .sparse import SparseSeg
from .util import Wrapper


class ConditionalInstSeg(Wrapper, Dataset):
    """
    Construct inputs (support image sparse annotations, query image)
    and targets (query dense annotations) for a conditional segmentation model.

    Args:
        supp_ds: the `Dataset` to load support labels
        qry_ds: the `Dataset` to load dense query labels as targets

    Note that this class assumes the two input datasets contain
    the same data in the same order.
    """

    def __init__(self, qry_ds, supp_ds, shot=1):
        super().__init__(qry_ds)
        self.qry_ds = qry_ds
        self.supp_ds = supp_ds
        self.shot = shot

    def __getitem__(self, idx):
        # n.b. one epoch loads each image in the query dataset once
        # load query image + target
        qry_im, qry_tgt, aux = self.qry_ds[idx]
        # load sparse input annotations
        supp = []
        supp_aux = []
        for i in range(self.shot):
            shot_im, shot_anno, shot_aux = self.supp_ds[idx]
            stacked_anno = np.zeros((self.num_classes, *shot_anno.shape), dtype=np.float32)
            for k in range(self.num_classes):
                stacked_anno[k].flat[shot_anno.flat == k] = 1
            supp.append((shot_im, stacked_anno))
            supp_aux.append(shot_aux)
        aux.update({'support': supp_aux})
        return qry_im, supp, qry_tgt, aux

    def __len__(self):
        return len(self.qry_ds)


class ConditionalSemSeg(Wrapper, Dataset):
    """
    Load inputs for conditional class segmentation network.

    Args:
        qry_ds: the `Dataset` to load query images and labels
        supp_ds: dict of `Dataset`s indexed by the semantic class
        from which they load data
    """
    def __init__(self, qry_ds, supp_ds, shot=1):
        super().__init__(qry_ds)
        self.qry_ds = qry_ds
        self.supp_datasets = supp_ds
        self.cls2idx = {list(ds.keep)[0]: i for i, ds in enumerate(self.supp_datasets)}
        self.shot = shot

    @property
    def num_classes(self):
        return 2  # 0 == negative, 1 == positive

    def __getitem__(self, idx):
        qry_im, qry_tgt, aux = self.qry_ds[idx]
        supp_ds = self.supp_datasets[self.cls2idx[aux['cls']]]
        supp = []
        supp_aux = []
        for i in range(self.shot):
            shot_im, shot_anno, shot_aux = supp_ds[np.random.randint(0, len(supp_ds))]
            stacked_anno = np.zeros((self.num_classes, *shot_anno.shape), dtype=np.float32)
            for k in range(self.num_classes):
                stacked_anno[k].flat[shot_anno.flat == k] = 1
            supp.append((shot_im, stacked_anno))
            supp_aux.append(shot_aux)
        aux.update({'support': supp_aux})
        return qry_im, supp, qry_tgt, aux

    def __len__(self):
        return len(self.qry_ds)
