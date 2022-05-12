import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple
from .autograd import conv2d, conv3d, firstconv2d, firstconv3d, abconv2d, abconv3d
import math
import pdb

conv2d = conv2d.apply
conv3d = conv3d.apply
firstconv2d = firstconv2d.apply
firstconv3d = firstconv3d.apply
abconv2d = abconv2d.apply
abconv3d = abconv3d.apply


class _ConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias, range, alpha, beta, downsample):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.range = range
        self.alpha = alpha
        self.beta = beta
        self.downsample = True
        if transposed:
            self.weight = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,range=None, alpha=1, beta=0, downsample=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, False, _pair(0), groups, bias, range, alpha, beta, downsample)

    def forward(self, input):
        if self.range:
            return firstconv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.range)
        elif self.alpha:
            return abconv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.alpha, self.beta, self.downsample)
        else:
            return conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.range, self.downsample)

class Conv3d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,range=None, alpha=1, beta=0, downsample=False):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, False, _triple(0), groups, bias, range, alpha, beta, downsample)

    def forward(self, input):
        if self.range:
            return firstconv3d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups,
                               self.range)
        elif self.alpha:
            return abconv3d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups,
                          self.alpha, self.beta, self.downsample)
        else:
            return conv3d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.downsample)
