import numpy as np

import torch
import torch.nn as nn

from .backbone import vgg16
from .fcn import Interpolator


class interactive_early(nn.Module):

    def __init__(self, num_classes, feat_dim=None):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim or 4096

        backbone = vgg16(is_caffe=True)
        # Modify conv1_1 to have 5 input channels
        # Init the weights in the new channels to the channel-wise mean
        # of the pre-trained conv1_1 weights
        old_conv1 = backbone.conv1_1.weight.data
        mean_conv1 = torch.mean(old_conv1, dim=1, keepdim=True)
        new_conv1 = nn.Conv2d(5, old_conv1.size(0), kernel_size=old_conv1.size(2), stride=1, padding=1)
        new_conv1.weight.data = torch.cat([old_conv1, mean_conv1, mean_conv1], dim=1)
        new_conv1.bias.data = backbone.conv1_1.bias.data
        backbone.conv1_1 = new_conv1
        self.encoder = backbone

        # classifier head
        self.head = nn.Conv2d(self.feat_dim, self.num_classes, 1)
        nn.init.constant_(self.head.weight, 0.)
        nn.init.constant_(self.head.bias, 0.)

        # bilinear interpolation for upsampling
        self.decoder = Interpolator(self.num_classes, 32, odd=False)
        # align output to input: see
        # https://github.com/BVLC/caffe/blob/master/python/caffe/coord_map.py
        self.encoder[0].padding = (81, 81)
        self.crop = 0


    def forward(self, x, anno):
        x = torch.cat((x, anno), dim=1)
        h, w = x.size()[-2:]
        x = self.encoder(x)
        x = self.head(x)
        x = self.decoder(x)
        x = x[..., self.crop:self.crop + h, self.crop:self.crop + w]
        return x
