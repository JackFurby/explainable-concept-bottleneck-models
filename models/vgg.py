"""
VGG Network modified from https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
New changes: add softmax layer + option for freezing lower layers except fc
"""
import os
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchexplain

from typing import Union, List, Dict, Any, cast

__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
    "x_to_c_model",
    "MLP",
    "End2EndModel"
]

model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-8a719046.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-19584684.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}

class End2EndModel(torch.nn.Module):
    def __init__(self, model1, model2, use_relu=False, use_sigmoid=False, n_class_attr=2):
        super(End2EndModel, self).__init__()
        self.first_model = model1
        self.sec_model = model2
        self.use_relu = use_relu
        self.use_sigmoid = use_sigmoid

    def forward_stage2(self, stage1_out):
        if self.use_relu:
            attr_outputs = [nn.ReLU()(o) for o in stage1_out]
            attr_outputs = torch.stack(attr_outputs)
        elif self.use_sigmoid:
            attr_outputs = [torch.nn.Sigmoid()(o) for o in stage1_out]
            attr_outputs = torch.stack(attr_outputs)
        else:
            attr_outputs = stage1_out

        stage2_inputs = attr_outputs
        #stage2_inputs = torch.cat(stage2_inputs, dim=1)
        all_out = [self.sec_model(stage2_inputs), stage1_out]
        return all_out

    def forward(self, x):
        outputs = self.first_model(x)
        return self.forward_stage2(outputs)


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, expand_dim, train=True):
        super(MLP, self).__init__()

        if train:
            self.lib = nn
        else:
            print("Using torchexplain")
            self.lib = torchexplain

        self.expand_dim = expand_dim
        if self.expand_dim:
            self.linear = self.lib.Linear(input_dim, expand_dim)
            self.activation = self.lib.ReLU(inplace=True)
            self.linear2 = self.lib.Linear(expand_dim, num_classes) #softmax is automatically handled by loss function
        else:
            self.linear = self.lib.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.linear(x)
        if hasattr(self, 'expand_dim') and self.expand_dim:
            x = self.activation(x)
            x = self.linear2(x)
        return x


def x_to_c_model(freeze, model, **kwargs):
    """
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        model (VGG object): Pytorch model
    """
    if freeze:  # only finetune fc layer
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    return model


class FC(nn.Module):

    def __init__(self, input_dim, output_dim, expand_dim, stddev=None, train=True, **kwargs):
        """
        Extend standard Torch Linear layer to include the option of expanding into 2 Linear layers
        """
        super(FC, self).__init__()

        if train:
            self.lib = nn
        else:
            self.lib = torchexplain

        self.expand_dim = expand_dim
        if self.expand_dim > 0:
            self.relu = self.lib.ReLU(inplace=True)
            self.fc_new = self.lib.Linear(input_dim, expand_dim)
            self.fc = self.lib.Linear(expand_dim, output_dim)
        else:
            self.fc = self.lib.Linear(input_dim, output_dim)
        if stddev:
            self.fc.stddev = stddev
            if expand_dim > 0:
                self.fc_new.stddev = stddev

    def forward(self, x):
        if self.expand_dim > 0:
            x = self.fc_new(x)
            x = self.relu(x)
        x = self.fc(x)
        return x


class VGG(nn.Module):
    def __init__(self, features=nn.Module, num_classes=200, n_attributes=0, bottleneck=False, expand_dim=0, connect_CY=False, init_weights=True, dropout=0.5, all_fc=nn.ModuleList(), train=True, **kwargs):
        """
        Args:
        num_classes: number of main task classes
        n_attributes: number of attributes to predict
        bottleneck: whether to make X -> A model
        expand_dim: if not 0, add an additional fc layer with expand_dim neurons
        """
        super().__init__()

        if train:
            self.lib = nn
        else:
            self.lib = torchexplain

        self.features = features
        self.n_attributes = n_attributes
        self.bottleneck = bottleneck
        self.avgpool = self.lib.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            self.lib.Linear(512 * 7 * 7, 4096),
            self.lib.ReLU(True),
            nn.Dropout(p=dropout),
            self.lib.Linear(4096, 4096),
            self.lib.ReLU(True),
            nn.Dropout(p=dropout),
        )
        self.all_fc_linear = self.lib.Linear(4096, n_attributes)

        if init_weights:
            self._initialize_weights()

        for m in self.modules():
            if isinstance(m, self.lib.Conv2d) or isinstance(m, self.lib.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, self.lib.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, X):
        x = self.features(X)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # N x 4096
        out = self.all_fc_linear(x)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, self.lib.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, self.lib.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, self.lib.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def load_partial_state_dict(self, state_dict):
        """
        If dimensions of the current model doesn't match the pretrained one (esp for fc layer), load whichever weights that match
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state or 'fc' in name:
                continue
            if isinstance(param, Parameter):
                param = param.data
            own_state[name].copy_(param)


def make_layers(cfg, batch_norm=False, train=True, alpha=1, beta=0, **kwargs):

    if train:
        lib = nn
        ab_kwargs = {}
    else:
        lib = torchexplain
        ab_kwargs = {"alpha": alpha, "beta": beta}

    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [lib.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = lib.Conv2d(in_channels, v, kernel_size=3, padding=1, **ab_kwargs)
            if batch_norm:
                layers += [conv2d, lib.BatchNorm2d(v), lib.ReLU(inplace=True)]
            else:
                layers += [conv2d, lib.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, train=True, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm, train=train, **kwargs), train=train, **kwargs)
    if pretrained:
        model.load_partial_state_dict(model_zoo.load_url(model_urls[arch]))
    return model


def vgg11(pretrained=False, progress=True, train=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11", "A", False, pretrained, progress, train=train, **kwargs)


def vgg11_bn(pretrained=False, progress=True, train=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11_bn", "A", True, pretrained, progress, train=train, **kwargs)


def vgg13(pretrained=False, progress=True, train=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13", "B", False, pretrained, progress, train=train, **kwargs)


def vgg13_bn(pretrained=False, progress=True, train=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13_bn", "B", True, pretrained, progress, train=train, **kwargs)


def vgg16(pretrained=False, progress=True, train=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16", "D", False, pretrained, progress, train=train, **kwargs)


def vgg16_bn(pretrained=False, progress=True, train=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16_bn", "D", True, pretrained, progress, train=train, **kwargs)


def vgg19(pretrained=False, progress=True, train=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19", "E", False, pretrained, progress, train=train, **kwargs)


def vgg19_bn(pretrained=False, progress=True, train=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19_bn", "E", True, pretrained, progress, train=train, **kwargs)
