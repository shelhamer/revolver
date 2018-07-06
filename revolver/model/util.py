import numpy as np

import torch
import torch.nn as nn


def update_state_dict(net, pretrained_dict):
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
