from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchvision import models

from .fcn import convolutionalize


def vgg16(is_caffe=True):
    """
    Load the VGG-16 net for use as a fully convolutional backbone.

    - cast to fully convolutional by converting `Linear` modules
    - name the same way as the original paper (for style and sanity)
    - load original Caffe weights (if requested)
    - decapitate last classifier layer
    - switch to ceiling mode for pooling like in Caffe

    Take
        is_caffe: flag for whether to load Caffe weights (default) or not
    """
    vgg16 = models.vgg16(pretrained=True)
    # cast into fully convolutional form (as list of layers)
    vgg16 = convolutionalize(list(vgg16.features) + list(vgg16.classifier),
                             (3, 224, 224))
    # name layers like the original paper
    names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5',
        'fc6', 'relu6', 'drop6', 'fc7', 'relu7', 'drop7', 'fc8']
    vgg16 = nn.Sequential(OrderedDict(zip(names, vgg16)))
    if is_caffe:
        # substitute original Caffe weights for improved fine-tuning accuracy
        # see https://github.com/jcjohnson/pytorch-vgg
        caffe_params = model_zoo.load_url('https://s3-us-west-2.amazonaws.com/'
                                          'jcjohns-models/vgg16-00b39a1b.pth')
        for new_p, old_p in zip(vgg16.parameters(), caffe_params.values()):
                new_p.data.copy_(old_p.view_as(new_p))
    # surgery: decapitate final classifier
    del vgg16._modules['fc8']  # note: risky use of private interface
    # surgery: keep fuller spatial dims by including incomplete pooling regions
    for m in vgg16.modules():
        if isinstance(m, nn.MaxPool2d):
            m.ceil_mode = True
    return vgg16
