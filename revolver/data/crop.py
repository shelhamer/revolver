import numpy as np
import math

from torch.utils.data import Dataset

from .util import Wrapper


class CropSeg(Wrapper, Dataset):
    """
    Crop images and targets loaded from wrapped dataset
    which must be a MaskSemSeg or MaskInstSeg
    Always include all of the foreground in the crop
    """

    def __init__(self, ds):
        super().__init__(ds)
        self.ds = ds

    def bbox(self, lbl):
        """
        Compute upper left and lower right coordinates
        of minimal bounding box containing foreground
        """
        positives = list(zip(*np.where(lbl==1)))
        positives.sort()
        x1, x2 = positives[0][0], positives[-1][0]
        positives.sort(key=lambda x: x[1])
        y1, y2 = positives[0][1], positives[-1][1]
        return (x1, y1, x2, y2)

    def enlarge(self, min_size, x1, x2):
        pad_x, remainder = divmod(64 - (x2 - x1), 2)
        x1 -= pad_x + remainder
        x2 += pad_x
        return x1, x2

    def __getitem__(self, idx):

        def safe_randint(low, high):
            if low == 0 and high == 0:
                return 0
            return np.random.randint(low, high)

        im, target, aux = self.ds[idx]
        # min size for crop is 224 - 80*2 = 64
        # n.b. this must be even
        ms = 64
        # pad image and target with 32px
        pad_im = np.full((im.shape[0] + ms, im.shape[1] + ms, 3), np.array(self.mean)*255., dtype=np.float32)
        pad_im[ms//2:-ms//2, ms//2:-ms//2, :] = im
        pad_target = np.pad(target, ((ms//2, ms//2), (ms//2, ms//2)), mode='constant', constant_values=0)
        # get instance bounding box, parameterized by upper left and lower right corners
        (bx1, by1, bx2, by2) = self.bbox(pad_target)
        # enlarge bounding box to minimum size
        x1, x2 = self.enlarge(ms, bx1, bx2)
        y1, y2 = self.enlarge(ms, by1, by2)
        # shift box and snap to instance without leaving padded image
        shift_x = safe_randint(max(-ms, -x1), min(ms, pad_target.shape[0] - x2))
        shift_y = safe_randint(max(-ms, -y1), min(ms, pad_target.shape[1] - y2))
        x1, x2 = min(x1 + shift_x, bx1), max(x2 + shift_x, bx2)
        y1, y2 = min(y1 + shift_y, by1), max(y2 + shift_y, by2)
        # scale box without leaving padded image
        x1 += safe_randint(max(-ms, -x1), 0)
        x2 += safe_randint(0, min(ms, pad_target.shape[0] - x2))
        y1 += safe_randint(max(-ms, -y1), 0)
        y2 += safe_randint(0, min(ms, pad_target.shape[1] - y2))
        # do the crop
        target = pad_target[x1:x2, y1:y2]
        im = pad_im[x1:x2, y1:y2, :]
        assert target.shape[0] >= ms and target.shape[1] >= ms, \
                'Crop is too small: {}'.format(target.shape)
        return im, target, aux

    def __len__(self):
        return len(self.slugs)
