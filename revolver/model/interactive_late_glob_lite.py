import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import vgg16
from .fcn import Interpolator, Downsampler
from .interactive_late_glob import interactive_late_glob


class interactive_late_glob_lite(interactive_late_glob):

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
