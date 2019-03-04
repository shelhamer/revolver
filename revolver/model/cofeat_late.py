import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .backbone import vgg16
from .fcn import Interpolator, Downsampler
from .dios_late import dios_late


class cofeat_late(dios_late):


    def __init__(self, num_classes, feat_dim=None):
        super().__init__(num_classes, feat_dim)
        self.register_buffer('z', torch.zeros(num_classes, self.feat_dim))
        self.register_buffer('num_z', torch.zeros(num_classes, 1).float())


    def forward(self, qry, supp):
        # extract guidance from support
        z = self.guide(supp)
        # segment query
        y = self.seg(qry, z)
        return y


    def guide(self, supp, update=False):
        z = Variable(self.z, requires_grad=False)  # for backprop through z
        num_z = Variable(self.num_z, requires_grad=False)
        if not update:
            # non-cumulative use resets accumulated guidance
            self.clear_guide()
            # clone guide and counter for one-time use and *do not update*
            z = z.clone()
            num_z = num_z.clone()
        for im, anno in supp:
            # encode support image
            feat = self.encoder(im)
            # cast annotations into feature masks
            anno = F.pad(anno, (0, 31, 0, 31), 'constant', 0)
            annos = torch.unbind(anno, dim=1)
            # only update guidance for given annotations
            active_idx = [i for i, a in enumerate(annos) if (a.sum() > 0).all()]
            if not active_idx:
                return z  # short-circuit for no annotations
            annos = [annos[i] for i in active_idx]
            annos = [self.anno_enc(a[None, ...]) for a in annos]
            annos = [a / (1e-6 + torch.sum(a.view(-1), dim=0)) for a in annos]
            # mask support by annotations
            z_shot = [self.mask_feat(feat, a) for a in annos]
            # global pool support +/- features
            z_shot = [torch.sum(z_.view(1, z_.size(1), -1), dim=2) for z_ in z_shot]
            z_shot = torch.cat(z_shot, dim=0)
            # accumulate guidance as running mean
            num_z[active_idx] += 1
            z[active_idx] += (z_shot - z[active_idx]) / num_z[active_idx]
        return z


    def seg(self, qry, z):
        h, w = qry.size()[-2:]
        qry = self.encoder(qry)

        # tile guidance across the query features
        feat_h, feat_w = qry.size()[-2:]
        z = [z_[None, ..., None, None] for z_ in torch.unbind(z, dim=0)]
        z = [z_.repeat(1, 1, feat_h, feat_w) for z_ in z]
        x = torch.cat([torch.cat((qry, z_), dim=1) for z_ in z], dim=0)

        # score by shared metric
        x = self.head(x)
        # interpolate and crop
        x = self.decoder(x)
        x = x[..., self.crop:self.crop + h, self.crop:self.crop + w]
        # unpack into annotation-wise channels
        x = x.permute(1, 0, 2, 3)
        return x


    def clear_guide(self):
        self.z.zero_()
        self.num_z.zero_()


    @property
    def way(self):
        # "way" == the no. of distinct annotation/output values
        return self.z.size(0)


    def set_way(self, k):
        self.z = self.z.new(k, self.feat_dim)
        self.num_z = self.num_z.new(k, 1)
        # new task, new guidance
        self.clear_guide()
