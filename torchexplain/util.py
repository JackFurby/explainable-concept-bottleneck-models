import torch
from torch import nn
import torch.nn.functional as F
from .autograd import avg_pool3d, avg_pool2d, add, scale
from . import Conv2d, Conv3d, BatchNorm2d, BatchNorm3d
avg_pool3d = avg_pool3d.apply
avg_pool2d = avg_pool2d.apply
# avg_pool1d = avg_pool1d.apply
add = add.apply
scale = scale.apply
from functools import partial
class Shortcut(nn.Module):
    def forward(self, *input):
        assert len(input) > 1
        return (torch.cat(tuple(input))).sum(dim=0,keepdim=True)

class Flatten(nn.Module):
    """
    Utility class for PyTorch models, please add this in place of any flattening you use in your model for use with LRP
    """
    def forward(self, input):
        return input.view(input.size(0), -1)

class Downsample(nn.Module):
    """
    Utility class for PyTorch models, please add this in place of any downsampling you use in your model for use with LRP.
    Models with shortcut connections (ResNET, DenseNET, Inception) may use this.
    """
    def __init__(self, in_, out_, mode="A", **kwargs):
        """
        :param out: The desired output size/shape for the downsample to result in.
        :param mode: A for Average Pooling followed by zero padding, B for Convolution followed by Batch Normalisation
        """
        super(Downsample, self).__init__()
        self.dim = kwargs.pop("dim", 2)
        self.kernel = kwargs.pop("kernel", 1)
        self.stride = kwargs.pop("stride", 1)

        if self.dim == 1:
            conv_layer = nn.Conv1d
            pool = F.avg_pool1d
            batch_layer = nn.BatchNorm1d
        elif self.dim == 2:
            conv_layer = Conv2d
            pool = avg_pool2d
            batch_layer = BatchNorm2d
        elif self.dim == 3:
            conv_layer = Conv3d
            pool = avg_pool3d
            batch_layer = BatchNorm3d



        def conv(x):
            return nn.Sequential(
                conv_layer(in_, out_, kernel_size=self.kernel, stride=self.stride, shortcut=True),
                batch_layer(out_)
            )

        self.mode = partial(basic, out=out_) if mode=="A" else conv
        return mode

    def forward(self, input):
        return self.mode(input)

class Add(nn.Module):
    def forward(self, *input):
        return add(*input)



class MinMaxScaler(nn.Module):
    def forward(self, input):
        return scale(input, input.min(), input.max() - input.min())
