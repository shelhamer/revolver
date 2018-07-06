import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _assert_no_grad


class CrossEntropyLoss2D(nn.CrossEntropyLoss):
    """
    Extend softmax + CE loss module to compute loss over spatial (2D) inputs.

    Take
        scores: the predictions with shape N x C x H x W
        target: the true target with shape N x 1 x H x W
    """

    def forward(self, scores, target):
        _assert_no_grad(target)
        if len(scores.size()) != 4:
            raise ValueError("Scores should have 4 dimensions, but has {}: {}".format(len(scores.size()), scores.size()))
        _, c, _, _ = scores.size()
        scores = scores.permute(0, 2, 3, 1).contiguous().view(-1, c)
        target = target.view(-1)
        return F.cross_entropy(scores, target, self.weight, self.size_average,
                               self.ignore_index, self.reduce)
