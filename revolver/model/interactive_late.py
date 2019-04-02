import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import vgg16
from .fcn import Interpolator, Downsampler


class interactive_late(nn.Module):

    def __init__(self, num_classes, feat_dim=None):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim or 4096

        # downsampling for annotation mask
        self.anno_enc = Downsampler(1, 32, odd=False)

        # fully conv encoder
        self.encoder = vgg16(is_caffe=True)
        del self.encoder[-3:]

        # classification head (including fc7 for compatibility with guidance)
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

        # bilinear interpolation for upsampling
        self.decoder = Interpolator(1, 32, odd=False)

        # align output to input: see
        # https://github.com/BVLC/caffe/blob/master/python/caffe/coord_map.py
        self.encoder[0].padding = (81, 81)
        self.crop = 0


    def forward(self, im, anno):
        h, w = im.size()[-2:]

        # Extract image features
        im = self.encoder(im)

        # Pre-process annotations and downsample them
        anno = F.pad(anno, (0, 31, 0, 31), 'constant', 0)
        annos = torch.unbind(anno, dim=1)
        annos = [self.anno_enc(a[None, ...]) for a in annos]
        annos = [a / (1e-6 + torch.sum(a.view(-1), dim=0)) for a in annos]

        # align image + mask, then mask features by annotations for guidance
        im_feats = self.mask_feat(im, annos[0], scale=False)
        guides = [self.mask_feat(im, a) for a in annos]

        # stack image-guidance pairs into batch dimension
        feat = torch.cat([torch.cat((im_feats, g), dim=1) for g in guides], dim=0)

        # score by shared metric
        scores = self.head(feat)
        # interpolate and crop
        upscores = self.decoder(scores)
        upscores = upscores[..., self.crop:self.crop + h, self.crop:self.crop + w]
        # unpack into annotation-wise channels
        upscores = upscores.permute(1, 0, 2, 3)
        return upscores


    def mask_feat(self, x, mask, scale=True):
        """
        Align spatial coordinates of feature and mask, crop feature, and
        multiply by mask if scale is True.

        Expect feature and mask to be N x C x H x W
        """
        # With input pad 81, fc6 crop offset is 0, so align upper lefts
        x_size, mask_size = x.size(), mask.size()
        if x_size[-2:] != mask_size[-2:]:
            raise ValueError("Shape mismatch. Feature is {}, but mask is {}".format(x_size, mask_size))
        m_dim = mask_size[-2:]
        x = x[:, :, :m_dim[0], :m_dim[1]]
        if scale:
            x = x * mask
        return x
