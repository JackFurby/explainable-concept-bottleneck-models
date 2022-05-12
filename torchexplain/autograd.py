import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
import torch.autograd as autograd
from torch.autograd import Function
import copy
from math import ceil, floor
import pdb
import matplotlib.pyplot as plt
import os
from . import epsilon


class conv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, downsample=False):
        # torch.Size([1, 64, 16, 56, 56])
        ctx.save_for_backward(input, weight, bias)
        ctx.hparam = [stride, padding, dilation, groups, downsample]
        return F.conv2d(input, weight, bias, stride,
                        padding, dilation, groups)
    @staticmethod
    def backward(ctx, grad_output):
        # input (N,C,D,H,W)
        # weights (F,C,DD,HH,WW)
        # output (N,F,Hd,Hh,Hw)
        input, weights, bias = ctx.saved_tensors
        stride, padding, dilation, groups, downsample = ctx.hparam
        weights = torch.clamp(weights, min=0)
        output = F.conv2d(input, weights, None, stride,
                        padding, dilation, groups)
        output[output==0] += epsilon
        norm_grad = grad_output / output
        # following EB special case, zero outputs result in relevance of 0 in grad
        if not epsilon:
            norm_grad[output==0] = 0

        if downsample:
            norm_grad = F.interpolate(norm_grad, size=input.shape[-2:], mode='bilinear')
        input.grad = torch.nn.grad.conv2d_input(input.shape, weights, norm_grad, stride=stride, padding=padding, groups=groups)
        if all(ctx.needs_input_grad):
            weights.grad = torch.nn.grad.conv2d_weight(input, weights.shape, norm_grad, stride=stride,
                                                padding=padding)
            if bias is not None:
                bias.grad = torch.nn.grad.conv2d_weight(input, bias.shape, norm_grad, stride=stride,
                                                padding=padding)
                return (input.grad * input), weights.grad, bias.grad, None, None, None, None, None
            return (input.grad * input), weights.grad, None, None, None, None, None, None
        return (input.grad * input), None, None, None, None, None, None, None


class conv3d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, downsample=False):
        # torch.Size([1, 64, 16, 56, 56])

        ctx.hparam = [stride, padding, dilation, groups, downsample]
        output = F.conv3d(input, weight, bias, stride,
                        padding, dilation, groups)
        ctx.save_for_backward(input, weight, bias)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        # input (N,C,D,H,W)
        # weights (F,C,DD,HH,WW)
        # output (N,F,Hd,Hh,Hw)
        input, weights, bias = ctx.saved_tensors
        stride, padding, dilation, groups, downsample= ctx.hparam
        weights = torch.clamp(weights, min=0)
        output = F.conv3d(input, weights, None, stride,
                          padding, dilation, groups)
        output[output == 0] += epsilon
        norm_grad = grad_output / output
        # following EB special case, zero outputs result in relevance of 0 in grad
        if not epsilon:
            norm_grad[output == 0] = 0

        if downsample:
            norm_grad = F.interpolate(norm_grad, size=input.shape[-3:], mode='trilinear')
        input.grad = torch.nn.grad.conv3d_input(input.shape, weights, norm_grad, stride=stride, padding=padding, groups=groups)
        if all(ctx.needs_input_grad):
            weights.grad = torch.nn.grad.conv3d_weight(input, weights.shape, norm_grad, stride=stride,
                                                       padding=padding)
            if bias is not None:
                bias.grad = torch.nn.grad.conv3d_weight(input, bias.shape, norm_grad, stride=stride,
                                                        padding=padding)
                return (input.grad * input), weights.grad, bias.grad, None, None, None, None, None
            return (input.grad * input), weights.grad, None, None, None, None, None, None
        return (input.grad * input), None, None, None, None, None, None, None


class firstconv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, range=None):
        ctx.save_for_backward(input, weight, bias)
        ctx.hparam = [stride, padding, dilation, groups, range]
        return F.conv2d(input, weight, bias, stride,
                        padding, dilation, groups)
    @staticmethod
    def backward(ctx, grad_output):
        input, weights, bias = ctx.saved_tensors
        stride, padding, dilation, groups, range = ctx.hparam
        pweights = nweights = weights
        pweights = torch.clamp(weights, min=0)
        nweights = torch.clamp(weights, max=0)
        lowest, highest = range
        linput = lowest + 0 * input.data
        hinput = highest + 0 * input.data
        output = F.conv2d(input, weights, None, stride,
                          padding, dilation, groups)
        pout = F.conv2d(linput, pweights, None, stride,
                          padding, dilation, groups)
        nout = F.conv2d(hinput, nweights, None, stride,
                          padding, dilation, groups)

        root_out = output - pout - nout
        root_out[root_out==0] += epsilon
        norm_grad = grad_output / root_out
        if not epsilon:
            norm_grad[root_out==0] = 0
        grad = torch.nn.grad.conv2d_input(input.shape, weights, norm_grad, stride=stride, padding=padding, groups=groups)
        pgrad = torch.nn.grad.conv2d_input(input.shape, pweights, norm_grad, stride=stride, padding=padding, groups=groups)
        ngrad = torch.nn.grad.conv2d_input(input.shape, nweights, norm_grad, stride=stride, padding=padding, groups=groups)
        if all(ctx.needs_input_grad):
            weights.grad = torch.nn.grad.conv2d_weight(input, weights.shape, norm_grad, stride=stride,
                                                       padding=padding)
            if bias is not None:
                bias.grad = torch.nn.grad.conv2d_weight(input, bias.shape, norm_grad, stride=stride,
                                                        padding=padding)
                return (grad * input - pgrad * linput - ngrad * hinput), weights.grad, bias.grad, None, None, None, None, None
            return (grad * input - pgrad * linput - ngrad * hinput), weights.grad, None, None, None, None, None, None
        return (grad * input - pgrad * linput - ngrad * hinput), None, None, None, None, None, None, None


class firstconv3d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, range=None):
        ctx.hparam = [stride, padding, dilation, groups, range]
        ctx.save_for_backward(input, weight, bias)
        return F.conv3d(input, weight, bias, stride,
                        padding, dilation, groups)
    @staticmethod
    def backward(ctx, grad_output):
        input, weights, bias = ctx.saved_tensors
        stride, padding, dilation, groups, range = ctx.hparam
        pweights = torch.clamp(weights, min=0)
        nweights = torch.clamp(weights, max=0)
        lowest, highest = range
        linput = lowest + 0 * input.data
        hinput = highest + 0 * input.data
        # output = F.conv3d(input, weights, None, stride,
                          # padding, dilation, groups)
        output = F.conv3d(input, weights, None, stride,
                          padding, dilation, groups)
        pout = F.conv3d(linput, pweights, None, stride,
                        padding, dilation, groups)
        nout = F.conv3d(hinput, nweights, None, stride,
                        padding, dilation, groups)


        root_out = output - pout - nout
        root_out[root_out == 0] += epsilon
        norm_grad = grad_output / root_out
        if not epsilon:
            norm_grad[root_out == 0] = 0
        grad = torch.nn.grad.conv3d_input(input.shape, weights, norm_grad, stride=stride, padding=padding, groups=groups)
        pgrad = torch.nn.grad.conv3d_input(input.shape, pweights, norm_grad, stride=stride, padding=padding, groups=groups)
        ngrad = torch.nn.grad.conv3d_input(input.shape, nweights, norm_grad, stride=stride, padding=padding, groups=groups)
        if all(ctx.needs_input_grad):
            weights.grad = torch.nn.grad.conv3d_weight(input, weights.shape, norm_grad, stride=stride,
                                                       padding=padding)
            if bias is not None:
                bias.grad = torch.nn.grad.conv3d_weight(input, bias.shape, norm_grad, stride=stride,
                                                        padding=padding)
                return (grad * input - pgrad * linput - ngrad * hinput), weights.grad, bias.grad, None, None, None, None, None
            return (grad * input - pgrad * linput - ngrad * hinput), weights.grad, None, None, None, None, None, None
        # testing
        return (grad * input - pgrad * linput - ngrad * hinput), None, None, None, None, None, None, None


class abconv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, beta=0, downsample=False):
        ctx.save_for_backward(input, weight, bias)
        ctx.hparam = [stride, padding, dilation, groups, alpha, beta, downsample]
        return F.conv2d(input, weight, bias, stride,
                        padding, dilation, groups)
    @staticmethod
    def backward(ctx, grad_output):
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        input, weights, bias = ctx.saved_tensors
        stride, padding, dilation, groups, alpha, beta, downsample = ctx.hparam

        pinput = torch.clamp(input, min=0)
        linput = torch.clamp(input, max=0)

        weights = torch.clamp(weights, min=0)
        pweights = nweights = weights
        pweights = torch.clamp(weights, min=0)
        nweights = torch.clamp(weights, max=0)

        pout = F.conv2d(pinput, pweights, None, stride,
                        padding, dilation, groups)
        nout = F.conv2d(linput, nweights, None, stride,
                        padding, dilation, groups)
        sum_out = pout + nout
        sum_out[sum_out == 0] += epsilon

        norm_grad = grad_output / sum_out
        if not epsilon:
            norm_grad[sum_out == 0] = 0

        #if downsample:
        #    norm_grad = F.interpolate(norm_grad, size=input.shape[-2:], mode='nearest')

        # norm_grad[sum_out == 0] = 0

        agrad = torch.nn.grad.conv2d_input(input.shape, pweights, norm_grad, stride=stride, padding=padding, groups=groups)
        agrad *= pinput
        bgrad = torch.nn.grad.conv2d_input(input.shape, nweights, norm_grad, stride=stride, padding=padding, groups=groups)
        bgrad *= linput

        grad = agrad + bgrad

        if beta:
            cpout = F.conv2d(pinput, nweights, None, stride,
                             padding, dilation, groups)
            cnout = F.conv2d(linput, pweights, None, stride,
                             padding, dilation, groups)
            csum_out = cpout + cnout
            csum_out[csum_out == 0] += epsilon
            c_grad = grad_output / csum_out
            if not epsilon:
                c_grad[csum_out == 0] = 0

            c_agrad = torch.nn.grad.conv2d_input(input.shape, nweights, c_grad, stride=stride, padding=padding, groups=groups)
            c_agrad *= pinput
            c_bgrad = torch.nn.grad.conv2d_input(input.shape, pweights, c_grad, stride=stride, padding=padding, groups=groups)
            c_bgrad *= linput

            c_grad = c_agrad + c_bgrad

            grad = alpha * grad - beta * c_grad
        if all(ctx.needs_input_grad):
            weights.grad = torch.nn.grad.conv2d_weight(input, weights.shape, grad, stride=stride, padding=padding)
            if bias is not None:
                bias.grad = torch.nn.grad.conv2d_weight(input, bias.shape, grad, stride=stride, padding=padding)
                return grad, weights.grad, bias.grad, None, None, None, None, None, None, None
            return grad, weights.grad, None, None, None, None, None, None, None, None
        return grad, None, None, None, None, None, None, None, None, None


class abconv3d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, beta=0, downsample=False):
        ctx.save_for_backward(input, weight, bias)
        ctx.hparam = [stride, padding, dilation, groups, alpha, beta,
                      downsample]
        return F.conv3d(input, weight, bias, stride,
                        padding, dilation, groups)
    @staticmethod
    def backward(ctx, grad_output):
        input, weights, bias = ctx.saved_tensors
        stride, padding, dilation, groups, alpha, beta, downsample = ctx.hparam

        pinput = torch.clamp(input, min=0)
        linput = torch.clamp(input, max=0)

        pweights = nweights = weights
        pweights = torch.clamp(weights, min=0)
        nweights = torch.clamp(weights, max=0)

        pout = F.conv3d(pinput, pweights, None, stride,
                        padding, dilation, groups)
        nout = F.conv3d(linput, nweights, None, stride,
                        padding, dilation, groups)
        sum_out = pout + nout
        sum_out[sum_out == 0] += epsilon

        norm_grad = grad_output / sum_out
        if not epsilon:
            norm_grad[sum_out == 0] = 0

        if downsample:
            norm_grad = F.interpolate(norm_grad, size=input.shape[-3:], mode='trilinear')

        agrad = torch.nn.grad.conv3d_input(input.shape, pweights, norm_grad, stride=1, padding=padding, groups=groups)
        agrad *= pinput
        bgrad = torch.nn.grad.conv3d_input(input.shape, nweights, norm_grad, stride=1, padding=padding, groups=groups)
        bgrad *= linput

        grad = agrad + bgrad

        if beta:
            cpout = F.conv3d(pinput, nweights, None, stride,
                             padding, dilation, groups)
            cnout = F.conv3d(linput, pweights, None, stride,
                             padding, dilation, groups)
            csum_out = cpout + cnout
            csum_out[csum_out == 0] += epsilon
            c_grad = grad_output / csum_out
            if not epsilon:
                c_grad[csum_out == 0] = 0

            c_agrad = torch.nn.grad.conv3d_input(input.shape, nweights, c_grad, stride=stride, padding=padding, groups=groups)
            c_agrad *= pinput
            c_bgrad = torch.nn.grad.conv3d_input(input.shape, pweights, c_grad, stride=stride, padding=padding, groups=groups)
            c_bgrad *= linput

            c_grad = c_agrad + c_bgrad

            grad = alpha * grad - beta * c_grad
        if all(ctx.needs_input_grad):
            weights.grad = torch.nn.grad.conv3d_weight(input, weights.shape, grad, stride=stride, padding=padding)
            if bias is not None:
                bias.grad = torch.nn.grad.conv3d_weight(input, bias.shape, grad, stride=stride, padding=padding)
                return grad, weights.grad, bias.grad, None, None, None, None, None, None, None
            return grad, weights.grad, None, None, None, None, None, None, None, None
        return grad, None, None, None, None, None, None, None, None, None


class avg_pool2d(Function):
    @staticmethod
    def forward(ctx, input, kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True):
        # if len(stride) == 1:
        #     # view 1d convolutions as 2d
        #     stride = (1,) + stride
        #     padding = (0,) + padding
        #     dilation = (1,) + dilation
        output = F.avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
        ctx.save_for_backward(input)
        ctx.hparams = [_pair(kernel_size), _pair(stride), _pair(padding), ceil_mode, count_include_pad]
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        kernel_size, stride, padding, ceil_mode, count_include_pad = ctx.hparams
        output = F.avg_pool2d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
        #if not padding:
        #    padding = 0
        N,C,H,W = input.shape
        n,c,h,w = output.shape
        k_h, k_w = kernel_size
        s_h, s_w = stride
        pad = []
        for p in _pair(padding):
            pad1 = p // 2
            pad2 = p - pad1
            pad = [pad1, pad2] + pad
        padding = tuple(pad)
        pad_input = F.pad(input, pad)
        padded_grad = torch.zeros_like(pad_input)
        for j in range(h-1):
            for k in range(w-1):
                Z = pad_input[:,:,j*s_h:j*s_h+k_h,k*s_w:k*s_w+k_w]
                Zs = Z.sum((2,3)).view(N,C,1,1)
                rr = grad_output[:,:,j:j+1,k:k+1].repeat((1,1,k_h,k_w))

                #breakpoint()

                zz = Z / Zs.repeat((1,1,k_h,k_w))
                zz[zz!=zz] = 0
                padded_grad[:,:,j*s_h:j*s_h+k_h,k*s_w:k*s_w+k_w] += rr * zz

        # following EB special case, zero outputs result in relevance of 0 in grad
        padding = list(padding)
        input.grad = padded_grad[...,padding[-2]:input.shape[2]+1-padding[-1],padding[-4]:input.shape[3]+1-padding[-3]]
        return (input.grad), None, None, None, None, None, None


class avg_pool3d(Function):
    @staticmethod
    def forward(ctx, input, kernel_size=1, stride=1, padding=0, ceil_mode=False, count_include_pad=True):
        # if len(stride) == 1:
        #     # view 1d convolutions as 2d
        #     stride = (1,) + stride
        #     padding = (0,) + padding
        #     dilation = (1,) + dilation
        output = F.avg_pool3d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
        ctx.save_for_backward(input)
        ctx.hparams = [_triple(kernel_size), _triple(stride), _triple(padding), ceil_mode, count_include_pad]
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        kernel_size, stride, padding, ceil_mode, count_include_pad = ctx.hparams
        output = F.avg_pool3d(input, kernel_size, stride, padding, ceil_mode, count_include_pad)
        N,C,D,H,W = input.shape
        n,c,d,h,w = output.shape
        k_d, k_h, k_w = kernel_size
        s_d, s_h, s_w = stride
        pad = []
        for p in _triple(padding):
            pad1 = p // 2
            pad2 = p - pad1
            pad = [pad1, pad2] + pad
        padding = tuple(pad)
        pad_input = F.pad(input, pad)
        padded_grad = torch.zeros_like(pad_input)
        for i in range(d):
            for j in range(h):
                for k in range(w):
                    Z = pad_input[:,:,i*s_d:i*s_d+k_d,j*s_h:j*s_h+k_h,k*s_w:k*s_w+k_w]
                    Zs = Z.sum((2,3,4)).view(N,C,1,1,1)
                    rr = grad_output[:,:,i:i+1,j:j+1,k:k+1].repeat((1,1,k_d,k_h,k_w))

                    zz = Z / Zs.repeat((1,1,k_d,k_h,k_w))
                    zz[zz!=zz] = 0
                    padded_grad[:,:,i*s_d:i*s_d+k_d,j*s_h:j*s_h+k_h,k*s_w:k*s_w+k_w] += rr * zz

        # following EB special case, zero outputs result in relevance of 0 in grad
        padding = list(padding)

        input.grad = padded_grad[...,padding[-2]:input.shape[2]+1-padding[-1],padding[-4]:input.shape[3]+1-padding[-3],padding[-6]:input.shape[4]+1-padding[-5]]
        return (input.grad), None, None, None, None, None, None

class max_pool2d(Function):
    @staticmethod
    def forward(ctx, input, kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False, return_indices=True):
        output, return_indices = F.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
        ctx.save_for_backward(input, output)
        ctx.hparams = [kernel_size, stride, padding, dilation, ceil_mode, return_indices]
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        kernel_size, stride, padding, dilation, ceil_mode, return_indices = ctx.hparams
        output, return_indices = F.max_pool2d(input, kernel_size, stride,
                                              padding, dilation, ceil_mode,
                                              True)
        N,C,H,W = input.shape
        n,c,h,w = output.shape
        k_h, k_w = kernel_size
        s_h, s_w = stride
        pad = []
        for p in _pair(padding):
            pad1 = p // 2
            pad2 = p - pad1
            pad = [pad1, pad2] + pad
        padding = tuple(pad)
        pad_input = F.pad(input, pad)
        padded_grad = torch.zeros_like(pad_input)
        for j in range(h):
            for k in range(w):
                # x = pad_input[:,:,j*s_h:j*s_h+k_h,k*s_w:k*s_w+k_w]
                # y = output[:,:,j:j+1,k:k+1].repeat((1,1,k_h,k_w))

                # Z = (x == y).float()
                # According to innvestigate, max pool should be treated as average pool in terms of relevance
                # This is likely due to the lossy downsampling that occurs in max pooling, causing a one takes all
                # strategy in terms of relevance distribution
                Z = pad_input[:,:,j*s_h:j*s_h+k_h,k*s_w:k*s_w+k_w]
                Zs = Z.sum((2,3)).view(N,C,1,1)

                rr = grad_output[:,:,j:j+1,k:k+1].repeat((1,1,k_h,k_w))
                zz = Z / Zs.repeat((1,1,k_h,k_w))
                zz[zz!=zz] = 0
                padded_grad[:,:,j*s_h:j*s_h+k_h,k*s_w:k*s_w+k_w] += rr * zz

        # following EB special case, zero outputs result in relevance of 0 in grad
        padding = list(padding)
        input.grad = padded_grad[...,padding[-2]:input.shape[2]+1-padding[-1],padding[-4]:input.shape[3]+1-padding[-3]]
        return (input.grad), None, None, None, None, None, None


class max_pool3d(Function):
    @staticmethod
    def forward(ctx, input, kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False, return_indices=True):
        output, return_indices = F.max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
        ctx.save_for_backward(input)
        ctx.hparams = [kernel_size, stride, padding, dilation, ceil_mode, return_indices]
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        kernel_size, stride, padding, dilation, ceil_mode, return_indices = ctx.hparams
        output, return_indices = F.max_pool3d(input, kernel_size, stride,
                                              padding, dilation, ceil_mode,
                                              True)
        N,C,D,H,W = input.shape
        n,c,d,h,w = output.shape
        k_d, k_h, k_w = kernel_size
        s_d, s_h, s_w = stride
        pad = []
        for p in _triple(padding):
            pad1 = p // 2
            pad2 = p - pad1
            pad = [pad1, pad2] + pad
        padding = tuple(pad)
        pad_input = F.pad(input, pad)
        padded_grad = torch.zeros_like(pad_input)
        for i in range(d):
            for j in range(h):
                for k in range(w):
                    # x = pad_input[:,:,i*s_d:i*s_d+k_d,j*s_h:j*s_h+k_h,k*s_w:k*s_w+k_w]
                    # y = output[:,:,i:i+1,j:j+1,k:k+1].repeat((1,1,k_d,k_h,k_w))

                    # Z = (x == y).float()
                    # According to innvestigate, max pool should be treated as average pool in terms of relevance
                    # This is likely due to the lossy downsampling that occurs in max pooling, causing a one takes all
                    # strategy in terms of relevance distribution
                    Z = pad_input[:,:,i*s_d:i*s_d+k_d,j*s_h:j*s_h+k_h,k*s_w:k*s_w+k_w]
                    Zs = Z.sum((2,3,4)).view(N,C,1,1,1)

                    rr = grad_output[:,:,i:i+1,j:j+1,k:k+1].repeat((1,1,k_d,k_h,k_w))
                    zz = Z / Zs.repeat((1,1,k_d,k_h,k_w))
                    zz[zz!=zz] = 0
                    padded_grad[:,:,i*s_d:i*s_d+k_d,j*s_h:j*s_h+k_h,k*s_w:k*s_w+k_w] += rr * zz

        # following EB special case, zero outputs result in relevance of 0 in grad
        padding = list(padding)
        input.grad = padded_grad[...,padding[-2]:input.shape[2]+1-padding[-1],padding[-4]:input.shape[3]+1-padding[-3],padding[-6]:input.shape[4]+1-padding[-5]]
        # pdb.set_trace()
        # f = input.mean((0,1,2)).cpu()
        # plt.imsave(f"{save_path}in.png",f)
        # f = input.grad.mean((0,1,2)).cpu()
        # plt.imsave(f"{save_path}ingrad.png",f)
        return (input.grad), None, None, None, None, None, None

class adaptive_avg_pool2d(Function):
    @staticmethod
    def forward(ctx, input, output_size):
        output = F.adaptive_avg_pool2d(input, output_size)
        ctx.save_for_backward(input)
        ctx.hparams = output_size
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        output_size = ctx.hparams
        #breakpoint()
        output = F.adaptive_avg_pool2d(input, output_size)
        # Reverse engineer params
        norm_grad = grad_output / output
        # following EB special case, zero outputs result in relevance of 0 in grad
        norm_grad[output == 0] = 0
        N, C, H, W = output.shape
        in_grad = torch.zeros_like(input)
        # The gradient of each element of the output can be distributed equally to each of the elements in the padded input pooling block
        for h in range(0,H,1):
            istartH = ceil(float(h*input.shape[2])/H)
            iendH = ceil(float((h+1)*input.shape[2])/H)
            kH = iendH - istartH
            for w in range(0,W,1):
                istartW = floor(float(w*input.shape[3])/W)
                iendW = ceil(float((w+1)*input.shape[3])/W)
                kW = iendW - istartW

                grad_delta = norm_grad[:,:,h,w] / kH / kW

                in_grad[:,:,istartH:iendH,istartW:iendW] += grad_delta.view(grad_delta.shape + (1,1))
        input.grad = in_grad
        return (input * input.grad), None, None, None, None, None

class adaptive_avg_pool3d(Function):
    @staticmethod
    def forward(ctx, input, output_size):
        output = F.adaptive_avg_pool3d(input, output_size)
        ctx.save_for_backward(input)
        ctx.hparams = output_size
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        output_size = ctx.hparams
        output = F.adaptive_avg_pool3d(input, output_size)
        # Reverse engineer params
        norm_grad = grad_output / output
        # following EB special case, zero outputs result in relevance of 0 in grad
        norm_grad[output == 0] = 0
        N, C, D, H, W = output.shape
        in_grad = torch.zeros_like(input)
        for d in range(0,D,1):
            istartD = floor(float(d*input.shape[2])/D)
            iendD = ceil(float((d+1)*input.shape[2])/D)
            kD = iendD - istartD
            for h in range(0,H,1):
                istartH = ceil(float(h*input.shape[3])/H)
                iendH = ceil(float((h+1)*input.shape[3])/H)
                kH = iendH - istartH
                for w in range(0,W,1):
                    istartW = floor(float(w*input.shape[4])/W)
                    iendW = ceil(float((w+1)*input.shape[4])/W)
                    kW = iendW - istartW

                    grad_delta = norm_grad[:,:,d,h,w] / kD / kH / kW

                    in_grad[:,:,istartD:iendD,istartH:iendH,istartW:iendW] += grad_delta.view(grad_delta.shape + (1,1,1))
        input.grad = in_grad
        return (input * input.grad), None, None, None, None, None


class linear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        ret = output
        ctx.save_for_backward(input, weight, bias)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        weight = torch.clamp(weight, min=0)
        output = input.matmul(weight.t())
        output[output==0] += epsilon
        norm_grad = grad_output / output
        if not epsilon:
            norm_grad[output==0] = 0
        # following EB special case, zero outputs result in relevance of 0 in grad
        # norm_grad[output==0] = 0
        input.grad = norm_grad.matmul(weight)
        return (input * input.grad), None, None


class threshold(Function):
    @staticmethod
    def forward(ctx, input, threshold, value, inplace=False):
        # TODO: Generalise!
        mask = torch.zeros_like(input)
        mask[input>0] += 1.
        ctx.save_for_backward(mask)
        return F.threshold(input,threshold,value, inplace)
    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.saved_tensors[0]
        return grad_output*mask, None, None, None


class scale(Function):
    @staticmethod
    def forward(ctx, input, min=0, range=1):
        output = torch.mul(torch.sub(input,min),1/range)
        ctx.save_for_backward(input, output)
        ctx.hparams = (min, range)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        inp, out = ctx.saved_tensors
        min, range = ctx.hparams
        return inp * grad_output/range*(inp-min)*out, None, None


class batch_norm(Function):
    @staticmethod
    def forward(ctx, input, running_mean, running_var, weight=None, bias=None,
               training=False, momentum=0.1, eps=1e-5):
        # Training not currently supported for torchtorchexplain
        # if training:
        #     size = input.size()
        #     #
        #     #
        #     # from operator import mul
        #     # from functools import reduce
        #     #
        #     #   if reduce(mul, size[2:], size[0]) == 1
        #     size_prods = size[0]
        #     for i in range(len(size) - 2):
        #         size_prods *= size[i + 2]
        #     if size_prods == 1:
        #         raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))
        output = torch.batch_norm(
            input, weight, bias, running_mean, running_var,
            training, momentum, eps, torch.backends.cudnn.enabled
        )
        ctx.save_for_backward(input, weight)
        ctx.hparams = (bias, running_mean, running_var, training, momentum, eps)
        return output
    @staticmethod
    def backward(ctx, grad_output):
        #|z|-rule as proposed in BatchNorm Decomposition for Deep Neural Network Optimisation
        input, weight= ctx.saved_tensors
        bias, running_mean, running_var, training, momentum, eps = ctx.hparams
        output = torch.batch_norm(
            input, weight, bias, running_mean, running_var,
            training, momentum, eps, torch.backends.cudnn.enabled
        )
        w_in = torch.zeros_like(input)
        b_in = torch.zeros_like(input)
        for w in range(len(weight)):
            w_in[:,w,...] = abs(weight[w] * input[:,w,...])
            b_in[:,w,...] = w_in[:,w,...] + abs(bias[w])
        out = w_in / b_in
        norm_grad = out * grad_output
        # f = (input.grad).cpu()
        # mdl = "sf"
        # plt.imsave(f"{save_path}{mdl}.png",f)
        return norm_grad, None, None, None, None, None, None, None

        #x1 = x2 = torch.zeros_like(input)
        #for b in range(bias.shape[0]):
        #    x2[:, b, ...] = output[:, b, ...] - bias[b]
        #for b in range(bias.shape[0]):
        #    x1[:,b,...] = input[:,b,...] - running_mean[b]
        #return (input * x2)*grad_output/(x1 * output + eps), None, None, None, None, None, None, None
        # return grad_output, None, None, None, None, None, None, None

class add(Function):
    @staticmethod
    def forward(ctx, *inputs):
        sum = torch.zeros_like(inputs[0])
        for input in inputs:
            sum = sum + input
        ctx.save_for_backward(*inputs)
        return sum
    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        sum = torch.sum(torch.stack(inputs),dim=0)
        norm_grad = grad_output / sum
        norm_grad[sum==0]=0
        for input in inputs:
            input.grad = norm_grad #/ len(inputs)

        return tuple([input * input.grad for input in inputs])
