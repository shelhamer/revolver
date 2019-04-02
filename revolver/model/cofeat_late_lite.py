import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import vgg16
from .fcn import Interpolator, Downsampler
from .cofeat_late import cofeat_late


class cofeat_late_lite(cofeat_late):


    def __init__(self, num_classes):
        super().__init__(num_classes, 256) # lite-headed: 256 vs. regular 4096

        # switch dim of fc6
        for k in list(self.encoder._modules)[-3:]:
            del self.encoder._modules[k]
        fc6 = [('fc6', nn.Conv2d(512, self.feat_dim, 7)),
            ('fc6_relu', nn.ReLU(inplace=True)),
            ('fc6_drop', nn.Dropout2d(p=0.5))]
        self.encoder._modules.update(fc6)
        # normal init new layer
        nn.init.normal_(self.encoder.fc6.weight, 0., .001)
        nn.init.constant_(self.encoder.fc6.bias, 0.)

        # classification head (including fc7 for compatibility with guidance)
        # overrride channel dimension since the pos/neg weights are not shared,
        # so we need three: query, positive, and negative.
        head = [('fc7', nn.Conv2d(self.feat_dim*2, self.feat_dim*2, 1)),
            ('fc7_relu', nn.ReLU(inplace=True)),
            ('fc7_drop', nn.Dropout2d(p=0.5)),
            ('score', nn.Conv2d(self.feat_dim*2, 1, 1))]
        self.head = nn.Sequential(OrderedDict(head))

        # normal init fc7
        nn.init.normal_(self.head.fc7.weight, 0., .001)
        # zero init score
        nn.init.constant_(self.head.score.weight, 0.)
        nn.init.constant_(self.head.score.bias, 0.)
