import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .util import Wrapper


class TransformData(Wrapper, Dataset):
    """
    Transform a dataset by registering a transform for every input and the
    target. Skip transformation by setting the transform to None.

    Take
        dataset: the `Dataset` to transform (which must be a `SegData`).
        input_transforms: list of `Transform`s for each input
        target_transform: `Transform` for the target image
    """

    def __init__(self, dataset, input_transforms=None, target_transform=None):
        super().__init__(dataset)
        self.ds = dataset
        self.input_transforms = input_transforms
        self.target_transform = target_transform
        # safety check
        num_inputs = len(self.ds[0]) - 2  # inputs are all but target, aux
        if len(self.input_transforms) != num_inputs:
            raise ValueError("The number of transformations {} does not match "
                             "the number of inputs {}".format(len(self.input_transforms), num_inputs))

    def __getitem__(self, idx):
        # extract data from inner dataset
        data = self.ds[idx]
        inputs, target, aux = data[:-2], data[-2], data[-1]
        inputs = list(inputs)  # for updating by transform
        for i, (input_, trans) in enumerate(zip(inputs, self.input_transforms)):
            if not isinstance(input_, list):
                inputs[i] = trans(input_) if trans is not None else input_
            else:
                inputs[i] = [[tr(in_) if tr is not None else in_
                              for in_, tr in zip(inp, trans)] for inp in input_]
        # transform target
        if self.target_transform is not None:
            target = self.target_transform(target)
        # repackage data
        return (*inputs, target, aux)

    def __len__(self):
        return len(self.ds)


class NpToTensor(object):
    """
    Convert `np.array` to `torch.Tensor`, but not like `ToTensor()`
    from `torchvision` because we don't rescale the values.
    """

    def __call__(self, arr):
        return torch.from_numpy(np.ascontiguousarray(arr))

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NpToIm(object):
    """
    Convert `np.array` to PIL `Image`, using mode appropriate for the
    number of channels.
    """

    def __call__(self, arr):
        if arr.shape[-1] == 1:
            return Image.fromarray(arr, mode='P')
        else:
            return Image.fromarray(arr, mode='RGB')

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ImToNp(object):
    """
    Convert PIL `Image` to `np.array`
    """

    def __call__(self, im):
        return np.array(im, dtype=np.uint8)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ImToCaffe(object):
    """
    Prepare image for input to Caffe-style network
     - permute RGB channels to BGR
     - subtract mean
     - swap axes to C x H x W order
    """

    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)[::-1].reshape(3, 1, 1) * 255.

    def __call__(self, im):
        im = im.astype(np.float32)[..., ::-1].transpose((2, 0, 1))
        im -= self.mean
        return im

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SegToTensor(object):
    """
    Convert `np.array` of discrete seg. labels to `torch.Tensor`.
    """

    def __call__(self, seg):
        seg = torch.from_numpy(seg.astype(np.uint8)).long()
        return seg

    def __repr__(self):
        return self.__class__.__name__ + '()'


class DilateMask(object):
    """
    Dilate a binary mask with filter of given size
    """

    def __init__(self, fs):
        self.fs = fs

    def __call__(self, arr):
        for i in range(arr.shape[0]):
            arr[i, ...] = np.array(Image.fromarray(arr[i, ...]).filter(ImageFilter.MaxFilter(self.fs)), dtype=arr.dtype)
        return arr

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ScaleMask(object):
    """
    Scale *binary* mask by given factor and re-center.

    Note: this casts the input to float for scaling.
    """

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, arr):
        arr = arr.astype(np.float32)
        return arr * self.factor - (self.factor / 2.)

    def __repr__(self):
        return self.__class__.__name__ + '()'
