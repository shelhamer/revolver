import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import vgg16
from .fcn import Interpolator, Downsampler
from .interactive_late import interactive_late


class cofeat_late_lite_unshared(interactive_late):

    def __init__(self, num_classes, feat_dim=None):
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
        head = [('fc7', nn.Conv2d(self.feat_dim*3, self.feat_dim*3, 1)),
            ('fc7_relu', nn.ReLU(inplace=True)),
            ('fc7_drop', nn.Dropout2d(p=0.5)),
            ('score', nn.Conv2d(self.feat_dim*3, num_classes, 1))]
        self.head = nn.Sequential(OrderedDict(head))

        # normal init fc7
        nn.init.normal_(self.head.fc7.weight, 0., .001)
        # zero init score
        nn.init.constant_(self.head.score.weight, 0.)
        nn.init.constant_(self.head.score.bias, 0.)

        # bilinear interpolation for upsampling
        # override channel dimension for unshared output
        self.decoder = Interpolator(num_classes, 32, odd=False)

    def forward(self, qry, supp):
        # query
        h, w = qry.size()[-2:]
        qry = self.encoder(qry)

        # encode support images
        supp_feats = [self.encoder(im) for im, _ in supp]

        # cast annotations into masks for feature maps
        pos_annos, neg_annos = [], []
        for _, anno in supp:
            anno = F.pad(anno, (0, 31, 0, 31), 'constant', 0)
            pos = anno[:, 0, ...].unsqueeze(1)
            neg = anno[:, 1, ...].unsqueeze(1)
            pos_anno = self.anno_enc(pos)
            neg_anno = self.anno_enc(neg)
            pos_anno = pos_anno / (1e-6 + torch.sum(pos_anno.view(-1), dim=0))
            neg_anno = neg_anno / (1e-6 + torch.sum(neg_anno.view(-1), dim=0))
            pos_annos.append(pos_anno)
            neg_annos.append(neg_anno)

        # mask support by annotations
        pos_feats = [self.mask_feat(f, a) for f, a in zip(supp_feats, pos_annos)]
        neg_feats = [self.mask_feat(f, a) for f, a in zip(supp_feats, neg_annos)]

        # global pool support +/- features and tile across query feature
        pos_vec = torch.cat([f.view(1, f.size(1), -1) for f in pos_feats], dim=2)
        neg_vec = torch.cat([f.view(1, f.size(1), -1) for f in neg_feats], dim=2)
        pos_glob = torch.sum(pos_vec, dim=2)  # 1 x C
        neg_glob = torch.sum(neg_vec, dim=2)
        pos_glob = pos_glob[..., None, None]  # 1 x C x 1 x 1
        neg_glob = neg_glob[..., None, None]
        # normalize by support size (mask is normalized by no. annotations)
        pos_glob = pos_glob.div_(len(supp))
        neg_glob = neg_glob.div_(len(supp))

        # Tile the pooled features across the image feature
        pos_glob = pos_glob.repeat(1, 1, qry.size(2), qry.size(3))
        neg_glob = neg_glob.repeat(1, 1, qry.size(2), qry.size(3))
        x = torch.cat([qry, neg_glob, pos_glob], dim=1)

        # inference from combined feature
        x = self.head(x)
        x = self.decoder(x)
        x = x[..., self.crop:self.crop + h, self.crop:self.crop + w]
        return x

