import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .backbone import vgg16
from .fcn import Interpolator, Downsampler
from .dios_late import dios_late


class dios_late_glob(dios_late):

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

        # global pool guidance + tile across image features
        guides = [torch.sum(g.view(g.size(0), g.size(1), -1), dim=2) for g in guides]  # N x C
        guides = [g[..., None, None] for g in guides]  # N x C x 1 x 1
        guides = [g.repeat(1, 1, im_feats.size(2), im_feats.size(3)) for g in guides]

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
