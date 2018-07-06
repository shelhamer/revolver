import numpy as np

from abc import abstractmethod
from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset

from .util import Wrapper


class SegData(Dataset):
    """
    Skeleton for loading datasets with input and target image pairs.

    Args:
        root_dir: path to dataset root dir
        split: specialized to the dataset, but usually train/val/test
    """

    classes = None

    # pixel statistics (RGB)
    mean = (0., 0., 0.)
    std = (1., 1., 1.)

    # special target value to exclude point from sampling, loss, and so on
    ignore_index = None

    def __init__(self, root_dir=None, split=None):
        self.root_dir = Path(root_dir)
        self.split = split

        self.slugs = self.load_slugs()

    @abstractmethod
    def load_slugs(self):
        pass

    @abstractmethod
    def slug_to_image_path(self, slug):
        pass

    def load_image(self, path):
        return np.array(Image.open(path), dtype=np.uint8)

    @abstractmethod
    def slug_to_annotation_path(self, slug):
        pass

    def load_annotation(self, path):
        return np.array(Image.open(path), dtype=np.uint8)

    @property
    def num_classes(self):
        return len(self.classes)

    def __getitem__(self, idx):
        slug = self.slugs[idx]
        im = self.load_image(self.slug_to_image_path(slug))
        target = self.load_annotation(self.slug_to_annotation_path(slug))
        # third return is reserved for auxiliary info dict
        return im, target, {}

    def __len__(self):
        return len(self.slugs)


class MaskSeg(Wrapper, Dataset):
    """
    Load data mask-wise instead of image-wise. Catalogue the masks in the
    wrapped dataset and return as binary masks with their class.

    n.b. Does not recognize 0/background and ignore index as masks, but does
    propagate the ignore values into the masks.

    Args:
        dataset: the `Dataset` to sample mask-wise (which must be a `SegData`).
        key: key for auxiliary dict to store ground truth value
    """

    def __init__(self, dataset, split='train'):
        super().__init__(dataset)
        self.ds = dataset
        self.slugs = self.load_slugs()
        self.split = split

    def load_slugs(self):
        slugs = []
        for i, (_, target, aux) in enumerate(self.ds):
            # for training, enumerate instances present in the frame
            if self.split == 'train':
                target = np.array(target, dtype=np.uint8)
                # take all values (but not background and ignore)
                values = np.unique(target)
                values = values[values != 0]
                values = values[values != self.ds.ignore_index]
            # for evaluation, enumerate instances present in the video
            else:
                values = self.load_instances(aux['vid'])
            slugs.extend([(i, {'vid': aux['vid'], 'inst': v}) for v in values])
        slugs.sort(key=lambda x: (x[1]['vid'], x[1]['inst']))
        return slugs

    @property
    def num_classes(self):
        return 2  # 0 == negative, 1 == positive

    def __getitem__(self, idx):
        # unpack slug and inner data
        inner_idx, this_aux = self.slugs[idx]
        im, target, aux = self.ds[inner_idx]
        # compose aux
        aux.update(this_aux)
        # make mask
        mask = np.zeros_like(target)
        mask[target == aux[self.key]] = 1
        mask[target == self.ds.ignore_index] = self.ds.ignore_index
        return im, mask, aux

    def __len__(self):
        return len(self.slugs)


class MaskSemSeg(Wrapper, Dataset):
    """
    Load data mask-wise instead of image-wise. Catalogue the masks in the
    wrapped dataset and return as binary masks with their class.
    n.b. Does not recognize 0/background and ignore index as masks, but does
    propagate the ignore values into the masks.
    Args:
        dataset: the `Dataset` to sample mask-wise (which must be a `SegData`).
    """

    def __init__(self, dataset):
        super().__init__(dataset)
        self.ds = dataset
        self.slugs = self.load_slugs()

    def load_slugs(self):
        slugs = []
        for i, (_, target, _) in enumerate(self.ds):
            target = np.array(target, dtype=np.uint8)
            # take all classes (but not background and ignore)
            classes = np.unique(target)
            classes = classes[classes != 0]
            classes = classes[classes != self.ds.ignore_index]
            slugs.extend([(i, {'cls': cls}) for cls in classes])
        return slugs

    @property
    def num_classes(self):
        return 2  # 0 == negative, 1 == positive

    def __getitem__(self, idx):
        # unpack slug and inner data
        inner_idx, this_aux = self.slugs[idx]
        im, target, aux = self.ds[inner_idx]
        # compose aux
        aux.update(this_aux)
        # make mask
        mask = np.zeros_like(target)
        mask[target == aux['cls']] = 1
        mask[target == self.ds.ignore_index] = self.ds.ignore_index
        return im, mask, aux

    def __len__(self):
        return len(self.slugs)


class MaskInstSeg(Wrapper, Dataset):
    """
    Load data mask-wise instead of image-wise. Catalogue the masks in the
    wrapped dataset and return as binary masks with their class and instance.

    n.b. Does not recognize 0/background and ignore index as masks, but does
    propagate the ignore values into the masks.

    Args:
        dataset: the `Dataset` to sample mask-wise (which must be a `SegData`).
    """

    def __init__(self, cls_dataset, inst_dataset):
        super().__init__(inst_dataset)
        self.cls_ds = cls_dataset
        self.inst_ds = inst_dataset
        self.slugs = self.load_slugs()

    def load_slugs(self):
        slugs = []
        for i, (cls, inst) in enumerate(zip(self.cls_ds, self.inst_ds)):
            inst_target = np.array(inst[1], dtype=np.uint8)
            # take all instances (but not background and ignore)
            instances = np.unique(inst_target)
            instances = instances[instances != 0]
            instances = instances[instances != self.inst_ds.ignore_index]
            cls_target = np.array(cls[1], dtype=np.uint8)
            # the class of every instance is the mode of the class target in the instance mask
            classes = [np.argmax(np.bincount(cls_target[inst_target == inst].flat)) for inst in instances]
            slugs.extend([(i, {'cls': cls, 'inst': inst}) for cls, inst in zip(classes, instances)])
        return slugs

    @property
    def num_classes(self):
        return 2  # 0 == negative, 1 == positive

    def __getitem__(self, idx):
        # unpack slug and inner data
        inner_idx, this_aux = self.slugs[idx]
        im, target, aux = self.inst_ds[inner_idx]
        # compose aux
        aux.update(this_aux)
        # make mask
        mask = np.zeros_like(target)
        mask[target == aux['inst']] = 1
        mask[target == self.inst_ds.ignore_index] = self.inst_ds.ignore_index
        return im, mask, aux

    def __len__(self):
        return len(self.slugs)
