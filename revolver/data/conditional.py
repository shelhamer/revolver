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


class ConditionalVideoSeg(Wrapper, Dataset):
    """
    Load inputs for conditional video segmentation network.

    Args:
        qry_ds: the `Dataset` to load query frames (a `MaskSemSeg`)
        supp_ds: dict of `Dataset`s keyed by sequence
        evaluation: if True, the support must be the first frame of the sequence and must contain all instances
    """
    def __init__(self, qry_ds, supp_ds, shot=1, evaluation=False):
        super().__init__(qry_ds)
        self.qry_ds = qry_ds
        self.supp_datasets = supp_ds
        self.shot = shot
        self.evaluation = evaluation

    @property
    def num_classes(self):
        return 2  # 0 == negative, 1 == positive

    def __getitem__(self, idx):
        # n.b. nothing prevents query and support from being the same frame. during evaluation, this will happen for the first frame.
        qry_im, qry_tgt, aux = self.qry_ds[idx]
        supp_ds = self.supp_datasets[aux['vid']]
        supp_indices = range(self.shot) if self.evaluation else np.random.choice(len(supp_ds), size=self.shot, replace=False)
        instances = [x for x in self.load_instances(aux['vid']) if x != 0]
        objects = instances
        # if training, sample a binary task
        if not self.evaluation:
            objects = [np.random.choice(instances)]
            for v in set(instances) - set(objects):
                qry_tgt[qry_tgt == v] = 0
            qry_tgt[qry_tgt == objects[0]] = 1
        objects = [0] + objects
        aux['num_cl'] = len(objects)

        supp = []
        supp_aux = []
        for supp_idx in supp_indices:
            shot_im, shot_anno, shot_aux = supp_ds[supp_idx]
            for v in set(instances) - set(objects):
                shot_anno[shot_anno == v] = 255
            # explode annotation into indicators
            # (float is necessary downstream for interpolation and such)
            stacked_anno = np.zeros((len(objects), *shot_anno.shape), dtype=np.float32)
            for i, k in enumerate(objects):
                stacked_anno[i].flat[shot_anno.flat == k] = 1
            supp.append((shot_im, stacked_anno))
            supp_aux.append(shot_aux)
        # TODO aux becomes a self-referential dict if you do this and wrap
        # in a transform dataset, don't know why!
        #aux.update({'support': supp_aux})
        return qry_im, supp, qry_tgt, aux

    def __len__(self):
        return len(self.qry_ds)

