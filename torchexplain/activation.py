import torch
from torch import nn
from .autograd import threshold
threshold = threshold.apply

class Threshold(nn.Module):
    def __init__(self, threshold, value, inplace):
        super(Threshold, self).__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace
    def forward(self, input):
        return threshold(input, self.threshold, self.value, self.inplace)

class ReLU(Threshold):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__(0., 0., inplace)
    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str