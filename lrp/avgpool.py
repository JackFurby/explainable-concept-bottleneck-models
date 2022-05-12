import torch
from .functional import avgpool2d

class AvgPool2d(torch.nn.AvgPool2d):
    def forward(self, input, explain=False, rule="epsilon", **kwargs):
        if not explain: return super(AvgPool2d, self).forward(input)
        return avgpool2d[rule](input, self.kernel_size, self.stride, self.padding)

    @classmethod
    def from_torch(cls, lin):
        module = cls(kernel_size=lin.kernel_size, stride=lin.stride, padding=lin.padding, ceil_mode=lin.ceil_mode, count_include_pad=lin.count_include_pad, divisor_override=lin.divisor_override)

        return module
