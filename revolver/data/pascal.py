import numpy as np

from abc import abstractmethod
from pathlib import Path

from .seg import SegData


class VOCSeg(SegData):
    """
    Load segmentation data in the style of PASCAL VOC.

    Args:
        root_dir: path to PASCAL VOC year dir
        split: {train,val,test}
    """

    classes = [
        '__background__',
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    # pixel statistics (RGB)
    mean = (0.48109378, 0.45752457, 0.40787054)
    std = (0.27363777, 0.26949592, 0.28480016)

    ignore_index = 255

    palette = np.array([
        [  0,   0,   0],
        [128,   0,   0],
        [  0, 128,   0],
        [128, 128,   0],
        [  0,   0, 128],
        [128,   0, 128],
        [  0, 128, 128],
        [128, 128, 128],
        [ 64,   0,   0],
        [192,   0,   0],
        [ 64, 128,   0],
        [192, 128,   0],
        [ 64,   0, 128],
        [192,   0, 128],
        [ 64, 128, 128],
        [192, 128, 128],
        [  0,  64,   0],
        [128,  64,   0],
        [  0, 192,   0],
        [128, 192,   0],
        [  0,  64, 128]], dtype=np.uint8)

    def __init__(self, **kwargs):
        kwargs['root_dir'] = kwargs.get('root_dir', 'data/voc2012')
        kwargs['split'] = kwargs.get('split', 'train')
        super().__init__(**kwargs)

    def load_slugs(self):
        listing = self.listing_path()
        with open(listing, 'r') as f:
            slugs = f.read().splitlines()
        return slugs

    def listing_path(self):
        return str(self.root_dir / 'ImageSets' / 'Segmentation' / '{}.txt'.format(self.split))

    def slug_to_image_path(self, slug):
        return str(self.root_dir / 'JPEGImages' / '{}.jpg'.format(slug))

    @abstractmethod
    def slug_to_annotation_path(self, slug):
        pass


class VOCSemSeg(VOCSeg):

    def slug_to_annotation_path(self, slug):
        return str(self.root_dir / 'SegmentationClass' / '{}.png'.format(slug))


class VOCInstSeg(VOCSeg):

    def slug_to_annotation_path(self, slug):
        return str(self.root_dir / 'SegmentationObject' / '{}.png'.format(slug))


class SBDDSemSeg(VOCSeg):
    """
    Load segmentation data in the style of SBDD, a further annotation of
    PASCAL VOC with semantic segmentation, instance segmentation, and contour
    annotations.

    n.b. The `trainaug` split is the union of VOC seg12train, SBD train, and SBD
    val minus its intersection with seg12val. It has 10,582 images/annotations.
    Where seg12train and SBDD intersect, the seg12train annotations are kept
    because they are more consistent. In order to make use of this split you
    have to convert the SBDD ground truth from mat to png in VOC format, copy
    the relevant images + annotations from VOC into the SBDD dataset dir, and
    include the trainaug.txt manifest of slugs in this split.

    Args:
        root_dir: path to sbdd root dir containing the `dataset` dir
        split: {train,val,test,trainaug}
        joint_transform: list of `Transform` to apply identically and jointly
                         to the image and target, such as horizontal flipping
        image_transform: list of `Transform`s for the input image
        target_transform: list of `Transform`s for the target image

    Note that joint transforms are done first so that tensor conversion can
    follow transformations done more simply on images/arrays, such as resizing.
    """

    def __init__(self, **kwargs):
        kwargs['root_dir'] = kwargs.get('root_dir', 'data/sbdd')
        super().__init__(**kwargs)

    def listing_path(self):
        return str(self.root_dir / 'dataset' / '{}.txt'.format(self.split))

    def slug_to_image_path(self, slug):
        return str(self.root_dir / 'dataset' / 'img' / '{}.jpg'.format(slug))

    def slug_to_annotation_path(self, slug):
        anno_format = 'mat' if self.split != 'trainaug' else 'png'
        return str(self.root_dir / 'dataset' / self.anno_type / '{}.{}'.format(slug, anno_format))

    def load_annotation(self, path):
        if self.split == 'trainaug':
            return super().load_annotation(path)
        else:
            import scipy.io
            mat = scipy.io.loadmat(path)['GT{}'.format(self.anno_type)][0]['Segmentation'][0]
            return mat.astype(np.uint8)

    @property
    def anno_type(self):
        return 'cls'


class SBDDInstSeg(SBDDSemSeg):

    @property
    def anno_type(self):
        return 'inst'
