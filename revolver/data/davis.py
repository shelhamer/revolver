import os
import numpy as np

from abc import abstractmethod
from pathlib import Path

from .seg import SegData


class DAVISInstSeg(SegData):
    """
    Load data from the DAVIS video segmentation dataset

    Args:
        root_dir: path to DAVIS year dir
        split: {train, val}
    """

    classes = ['__background__', 'foreground']

    # pixel statistics
    mean = (0.48109378, 0.45752457, 0.40787054)
    std = (0.27363777, 0.26949592, 0.28480016)

    ignore_index = 255 # n.b. does not actually occur in the data

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
        kwargs['root_dir'] = kwargs.get('root_dir', 'data/davis')
        kwargs['split'] = kwargs.get('split', 'train')
        super().__init__(**kwargs)

    def load_videos(self):
        listing = self.listing_path()
        with open(listing, 'r') as f:
            videos = f.read().splitlines()
        return videos

    def load_instances(self, video):
        # all labeled instances in the video must appear in the first frame
        target = self.load_annotation(self.slug_to_annotation_path((video,'00000')))
        instances = [x for x in np.unique(target) if x != self.ignore_index]
        return instances

    def load_slugs(self):
        videos = self.load_videos()
        slugs = []
        for vid in videos:
            # slug consists of (video name, frame index) and store video name in aux for downstream datasets
            slugs.extend([((vid, frm[:-4]), {'vid': vid, 'frm': frm[:-4]}) for frm in sorted(os.listdir(str(self.root_dir / 'JPEGImages' / '480p' / vid)), key=lambda x: int(x[:-4]))])
        return slugs

    def listing_path(self):
        return str(self.root_dir / 'ImageSets' / '2017' / '{}.txt'.format(self.split))

    def slug_to_image_path(self, slug):
        return str(self.root_dir / 'JPEGImages' / '480p' / '{}/{}.jpg'.format(*slug))

    def slug_to_annotation_path(self, slug):
        return str(self.root_dir / 'Annotations' / '480p' / '{}/{}.png'.format(*slug))

    def __getitem__(self, idx):
        # n.b. we override the SegData method because we keep track of
        # the video in the aux dict
        slug, aux = self.slugs[idx]
        im = self.load_image(self.slug_to_image_path(slug))
        target = self.load_annotation(self.slug_to_annotation_path(slug))
        return im, target, aux


