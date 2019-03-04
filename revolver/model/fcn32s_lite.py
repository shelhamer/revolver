import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn

from .backbone import vgg16
from .fcn import Interpolator
from .fcn32s import fcn32s


class fcn32s_lite(fcn32s):
    """
    FCN-32s-lite: fully convolutional network with VGG-16 backbone,
    light-headed edition with only 256 channels after conv5.
    """

    def __init__(self, num_classes):
        super().__init__(num_classes, 256) # lite-headed: 256 vs. regular 4096

        # encoder: VGG16 with different channel dim.
        for k in list(self.encoder._modules)[-6:]:
            del self.encoder._modules[k]
        fc6 = [('fc6', nn.Conv2d(512, self.feat_dim, 7)),
            ('fc6_relu', nn.ReLU(inplace=True)),
            ('fc6_drop', nn.Dropout2d(p=0.5))]
        self.encoder._modules.update(fc6)
        fc7 = [('fc7', nn.Conv2d(self.feat_dim, self.feat_dim, 1)),
            ('fc7_relu', nn.ReLU(inplace=True)),
            ('fc7_drop', nn.Dropout2d(p=0.5))]
        self.encoder._modules.update(fc7)
        # normal init new layers
        for m in (self.encoder.fc6, self.encoder.fc7):
            m.weight.data.normal_(0, .001)
            m.bias.data.zero_()
