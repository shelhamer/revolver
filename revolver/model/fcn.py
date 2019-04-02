import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def convolutionalize(modules, input_size):
    """
    Recast `modules` into fully convolutional form.

    The conversion transfers weights and infers kernel sizes from the
    `input_size` and modules' action on it.

    n.b. This only handles the conversion of linear/fully-connected modules,
    although other module types could require conversion for correctness.

    """
    fully_conv_modules = []
    x = torch.zeros((1, ) + input_size)
    for m in modules:
        if isinstance(m, nn.Linear):
            n = nn.Conv2d(x.size(1), m.weight.size(0), kernel_size=(x.size(2), x.size(3)))
            n.weight.data.view(-1).copy_(m.weight.data.view(-1))
            n.bias.data.view(-1).copy_(m.bias.data.view(-1))
            m = n
        fully_conv_modules.append(m)
        x = m(x)
    return fully_conv_modules


def bilinear_kernel(size, normalize=False):
    """
    Make a 2D bilinear kernel suitable for upsampling/downsampling with
    normalize=False/True. The kernel is size x size square.

    Take
        size: kernel size (square)
        normalize: whether kernel sums to 1 (True) or not

    Give
        kernel: np.array with bilinear kernel coefficient
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    kernel = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    if normalize:
        kernel /= kernel.sum()
    return kernel


class Interpolator(nn.Module):
    """
    Interpolate by de/up/backward convolution with a bilinear kernel.

    Take
        channel_dim: the input channel dimension
        rate: upsampling rate, that is 4 -> 4x upsampling
        odd: the kernel parity, which is too much to explain here for now, but
             will be handled automagically in the future, promise.
        normalize: whether kernel sums to 1
    """
    def __init__(self, channel_dim, rate, odd=True, normalize=False):
        super().__init__()
        self.rate = rate
        ksize = rate * 2
        if odd:
            ksize -= 1
        # set weights to within-channel bilinear interpolation
        kernel = torch.from_numpy(bilinear_kernel(ksize, normalize))
        weight = torch.zeros(channel_dim, channel_dim, ksize, ksize)
        for k in range(channel_dim):
            weight[k, k] = kernel
        # fix weights
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        # no groups (for speed with current pytorch impl.) and no bias
        return F.conv_transpose2d(x, self.weight, stride=self.rate)


class Downsampler(Interpolator):
    '''
    Downsample with a normalized bilinear kernel.
    '''
    def __init__(self, channel_dim, rate, odd=True):
        super().__init__(channel_dim, rate, odd, True)

    def forward(self, x):
        return F.conv2d(x, self.weight, stride=self.rate)
