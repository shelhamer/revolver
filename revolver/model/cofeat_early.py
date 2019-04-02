import numpy as np
from collections import OrderedDict
import copy

import torch
import torch.nn as nn

from .backbone import vgg16
from .fcn import Interpolator


class cofeat_early(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        feat_dim = 256

        # make separate fully conv encoders for support and query
        backbone = vgg16(is_caffe=True)
        del backbone[-6:]
        fc6 = [('fc6', nn.Conv2d(512, feat_dim, 7)),
            ('fc6_relu', nn.ReLU(inplace=True)),
            ('fc6_drop', nn.Dropout2d(p=0.5))]
        for n, m in fc6:
            setattr(backbone, n, m)
        supp_backbone = copy.deepcopy(backbone)
        qry_backbone = copy.deepcopy(backbone)

        # Modify conv1_1 of conditioning branch to have 5 input channels
        # Init the weights in the new channels to the channel-wise mean
        # of the pre-trained conv1_1 weights
        old_conv1 = supp_backbone._modules['conv1_1'].weight.data
        mean_conv1 = torch.mean(old_conv1, dim=1, keepdim=True)
        new_conv1 = nn.Conv2d(5, old_conv1.size(0), kernel_size=old_conv1.size(2), stride=1, padding=1)
        new_conv1.weight.data = torch.cat([old_conv1, mean_conv1, mean_conv1], dim=1)
        new_conv1.bias.data = supp_backbone._modules['conv1_1'].bias.data
        supp_backbone.conv1_1 = new_conv1
        self.supp_encoder = supp_backbone
        self.qry_encoder = qry_backbone

        # classifier head
        fc7 = [('fc7', nn.Conv2d(feat_dim*2, feat_dim, 1)),
            ('fc7_relu', nn.ReLU(inplace=True)),
            ('fc7_drop', nn.Dropout2d(p=0.5))]
        score = [('score', nn.Conv2d(feat_dim, num_classes, 1))]
        self.head = nn.Sequential(OrderedDict(fc7 + score))

        # FC6 and FC7 should be init with random Gaussian weights
        # Score layer should be zero
        for n, m in self.named_modules():
            if 'fc' in n and isinstance(n, nn.Conv2d):
                nn.init.normal_(m.weight, 0., .001)
            elif 'score' in n:
                nn.init.constant_(m.weight, 0.)
                nn.init.constant_(m.bias, 0.)

        # bilinear interpolation for upsampling
        self.decoder = Interpolator(num_classes, 32, odd=False)

        # align output to input: see
        # https://github.com/BVLC/caffe/blob/master/python/caffe/coord_map.py
        self.supp_encoder[0].padding = (81, 81)
        self.qry_encoder[0].padding = (81, 81)
        self.crop = 0

    def forward(self, qry, supp):
        # query
        h, w = qry.size()[-2:]
        qry = self.qry_encoder(qry)
        # support: concat image + annotation then encode
        supp = [torch.cat(s, dim=1) for s in supp]
        supp = [self.supp_encoder(s) for s in supp]
        # global pool support feature and tile it across query feature
        supp = torch.cat([f.view(1, f.size(1), -1) for f in supp], dim=2)
        supp = torch.mean(supp, dim=2)
        supp = supp[..., None, None]
        supp = supp.repeat(1, 1, qry.size(2), qry.size(3))
        # note: concat support first, unlike others. TODO(shelhamer) switch?
        x = torch.cat([supp, qry], dim=1)
        # inference from combined feature
        x = self.head(x)
        x = self.decoder(x)
        x = x[..., self.crop:self.crop + h, self.crop:self.crop + w]
        return x
