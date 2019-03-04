from pathlib import Path

from .seg import SegData


class COCO(SegData):
    """
    Load semantic segmentation data from COCO converted to VOC style.

    Take
        root_dir: path to COCO year dir
        split: {train,val}
    """

    classes =  ['__background__', 'person', 'bicycle', 'car', 'motorcycle',
            'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
            'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
            'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    # pixel statistics (RGB)
    mean = (0.48109378, 0.45752457, 0.40787054)
    std = (0.27363777, 0.26949592, 0.28480016)

    # reserved target value to exclude from loss, evaluation, ...
    ignore_index = 255

    def __init__(self, **kwargs):
        kwargs['root_dir'] = kwargs.get('root_dir', 'data/coco2017')
        super().__init__(**kwargs)

    def load_slugs(self):
        with open(self.listing_path(), 'r') as f:
            slugs = f.read().splitlines()
        return slugs

    def listing_path(self):
        return self.root_dir / f'{self.split}.txt'

    def slug_to_image_path(self, slug):
        return self.root_dir / f'{self.split}2017' / f'{slug}.jpg'

    def slug_to_annotation_path(self, slug):
        return (self.root_dir / f'annotations/seg_{self.split}2017'
                / f'{slug}.png')
