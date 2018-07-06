from collections import OrderedDict

import numpy as np


def fast_hist(a, b, n):
    """
    Fast 2D histogram by linearizing.
    """
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


class SegScorer(object):
    """
    Score semantic segmentation metrics by accumulating histogram
    of outputs and targets.

    - overall pixel accuracy
    - per-class accuracy
    - per-class intersection-over-union (IU)
    - frequency weighted IU

    n.b. mean IU is the standard single number summary of segmentation quality.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((self.num_classes, self.num_classes))
        # need to measure bg-bg intersection class-wise for heldout evaluation
        self.bg = np.zeros((self.num_classes,))

    def update(self, output, target, aux):
        if 'cls' in aux:
            # binary tasks on masks: map positive to true class
            output[output == 1] = aux['cls']
            target[target == 1] = aux['cls']
        elif 'mapping' in aux and len(set(aux['mapping'].values())) == 2:
            # foreground-background: assign foreground to true class(es)
            # for class-wise scoring by taking product of output and full truth
            target = aux['full_target']
            output *= target
        # regular scoring
        hist = fast_hist(target.flat, output.flat, self.num_classes)
        self.hist +=  hist
        # background measurement for class-wise scoring
        if 'cls' in aux:
            self.bg[aux['cls']] += hist[0, 0]

    def score(self):
        scores = OrderedDict()
        # overall accuracy
        scores['all_acc'] = np.diag(self.hist).sum() / self.hist.sum()
        with np.errstate(divide='ignore', invalid='ignore'):  # missing classes are ok
            # per-class accuracy
            scores['acc'] = np.diag(self.hist) / self.hist.sum(1)
            # per-class IU
            scores['iu'] = np.diag(self.hist) / (self.hist.sum(1) + self.hist.sum(0) - np.diag(self.hist))
        # frequency-weighted IU
        freq = self.hist.sum(1) / self.hist.sum()
        scores['freq_iu'] = (freq[freq > 0] * scores['iu'][freq > 0]).sum()
        # binary IU (neg: 0, pos >=1)
        binary_hist = np.array((self.hist[0, 0], self.hist[0, 1:].sum(),
            self.hist[1:, 0].sum(), self.hist[1:, 1:].sum())).reshape((2, 2))
        scores['bin_iu'] = np.diag(binary_hist) / (binary_hist.sum(1) + binary_hist.sum(0) - np.diag(binary_hist))
        # mean IU w/o background as in Shaban et al. BMVC'17
        scores['nobg_iu'] = scores['iu'][1:]
        # IU for positive alone
        scores['pos_iu'] = scores['bin_iu'][1]
        return scores

    def save(self, path):
        np.savez(path, hist=self.hist, bg=self.bg)
