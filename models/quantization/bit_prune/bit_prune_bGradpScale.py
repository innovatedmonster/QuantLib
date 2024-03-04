#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 令bit是连续可导，且使用plain scale而不是lsq更新scale
# 尝试使用

from __future__ import print_function, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math

__all__ = ['BPConv2d', "Round", "FunLSQ"]

class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = torch.clone(grad_output)
        return grad_input

class FunLSQ(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, per_channel=False):
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp, per_channel
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)             
            alpha = torch.broadcast_to(alpha, weight.size())
            w_q = Round.apply(torch.div(weight, alpha)).clamp(Qn, Qp)
            w_q = w_q * alpha
            w_q = torch.transpose(w_q, 0, 1)
            w_q = w_q.contiguous().view(sizes)
        else:
            w_q = Round.apply(torch.div(weight, alpha)).clamp(Qn, Qp)
            w_q = w_q * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp, per_channel = ctx.other
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            q_w = weight / alpha
            q_w = torch.transpose(q_w, 0, 1)
            q_w = q_w.contiguous().view(sizes)
        else:
            q_w = weight / alpha
        smaller = (q_w < Qn).float()
        bigger = (q_w > Qp).float()
        between = 1.0 - smaller - bigger  # 得到位于量化区间的index
        if per_channel:
            grad_alpha = ((smaller * Qn + bigger * Qp + between * Round.apply(q_w) - between * q_w)*grad_weight * g)
            grad_alpha = grad_alpha.contiguous().view(grad_alpha.size()[0], -1).sum(dim=1)
        else:
            grad_alpha = ((smaller * Qn + bigger * Qp +
                           between * Round.apply(q_w) - between * q_w)*grad_weight * g).sum().unsqueeze(dim=0)# grad must be shape[1] tensor
        grad_weight = between * grad_weight

        return grad_weight, None, None, None, None, None


class LSQActQuantizer(nn.Module):
    def __init__(self, bits, qtype="quint", quant=False, observer=False, learning=False):
        super(LSQActQuantizer, self).__init__()
        self.bits = bits
        self.qtype = qtype
        self.quant = quant
        self.observer = observer
        self.learning = learning
        assert self.bits != 1, "LSQ don't support binary quantization"
        assert self.qtype in ("qint", "quint"), "qtype just support qint or quint"
        self.calQnQp()

        self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.scale_1 = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.alpha_bit = torch.nn.Parameter(torch.ones(1), requires_grad=True)#added, to decide bitwidth
        # self.alpha_bit = torch.nn.Parameter(torch.zeros(1), requires_grad=True)#added, to decide bitwidth
        self.grad_factor = 1.0
        self.grad_factor_1 = 1.0
        self.observer_init = torch.tensor(1, dtype=torch.int8)

        #added self.max, self.min
        min = torch.tensor([], requires_grad=False)
        max = torch.tensor([], requires_grad=False)
        self.register_buffer('min', min)
        self.register_buffer('max', max)

    def calQnQp(self):
        if self.qtype == "quint":
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2 ** self.bits - 1
            self.Qn_1 = 0
            self.Qp_1 = 2 ** (self.bits-1) - 1
            # self.Qp_1 = 2 ** (bits+1) - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = - 2 ** (self.bits - 1)
            self.Qp = 2 ** (self.bits - 1) - 1
            self.Qn_1 = - 2 ** (self.bits-1-1)
            self.Qp_1 = 2 ** (self.bits-1-1) - 1
            # self.Qn_1 = - 2 ** (bits-1+1)
            # self.Qp_1 = 2 ** (bits-1+1) - 1
        
        #added, to maintain 1 bit's calculation runnable
        if self.bits-1 == 1:
            self.Qn = 0
            self.Qp = 2 ** self.bits - 1
            self.Qn_1 = 0
            self.Qp_1 = 2 ** (self.bits-1) - 1

    def forward(self, x):
        if not self.quant:
            return x
        #bug here，not support 1 bit which makes self.Qp equals 0 so that division error
        self.grad_factor = 1.0 / math.sqrt(x.numel() * self.Qp)
        self.grad_factor_1 = 1.0 / math.sqrt(x.numel() * self.Qp_1)
        if self.observer:
            if self.observer_init == 1:
                self.scale.data[0] = torch.mean(torch.abs(x.detach()))*2/math.sqrt(self.Qp)
                self.scale_1.data[0] = torch.mean(torch.abs(x.detach()))*2/math.sqrt(self.Qp_1)
                self.observer_init = 0
            else:
                self.scale.data[0] = 0.9*self.scale.data[0] + 0.1*torch.mean(torch.abs(x.detach()))*2/math.sqrt(self.Qp)
                self.scale_1.data[0] = 0.9*self.scale_1.data[0] + 0.1*torch.mean(torch.abs(x.detach()))*2/math.sqrt(self.Qp_1)
        if self.observer or self.learning:
            # x = FunLSQ.apply(x, self.scale, self.grad_factor, self.Qn, self.Qp)
            self.calcScale(x)
            x = (1-self.alpha_bit) * FunLSQ.apply(x, self.scale_1, self.grad_factor_1, self.Qn_1, self.Qp_1) + \
            self.alpha_bit * FunLSQ.apply(x, self.scale, self.grad_factor, self.Qn, self.Qp)
            # x = (1-self.alpha_bit.data[0]) * FunLSQ.apply(x, self.scale, self.grad_factor, self.Qn, self.Qp) + \
            # self.alpha_bit.data[0] * FunLSQ.apply(x, self.scale, self.grad_factor, self.Qn_1, self.Qp_1)

        # print('alpha and bit are ', self.alpha_bit.data[0], self.bits)
        
        #added, for bit prune
        if self.alpha_bit.data[0] < 1e-4:
            self.bits = self.bits - 1
            self.alpha_bit.data[0] = 1.0
            self.calQnQp()
        # if self.alpha_bit.data[0] >= 1.0:
        #     self.bits = self.bits + 1
        #     self.alpha_bit.data[0] = 0.0
        return x
    
    def calcScale(self, tensor):
        if self.max.nelement() == 0 or self.max.data < tensor.max().data:
            self.max.data = tensor.max().data
        self.max.clamp_(min=0)

        if self.min.nelement() == 0 or self.min.data > tensor.min().data:
            self.min.data = tensor.min().data
        self.min.clamp_(max=0)

        self.scale.data[0] = (self.max - self.min) / (self.Qp - self.Qn)
        self.scale_1.data[0] = (self.max - self.min) / (self.Qp_1 - self.Qn_1)


class LSQWeightQuantizer(nn.Module):
    def __init__(self, bits, qtype="qint", per_channel=False, quant=False, observer=False, learning=False):
        super(LSQWeightQuantizer, self).__init__()
        self.bits = bits
        self.qtype = qtype
        self.quant = quant
        self.observer = observer
        self.learning = learning

        assert self.bits != 1, "LSQ don't support binary quantization"
        assert self.qtype in ("qint", "quint"), "qtype just support qint or quint"
        self.calQnQp()

        self.per_channel = per_channel
        self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.scale_1 = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.alpha_bit = torch.nn.Parameter(torch.ones(1), requires_grad=True)#added, to decide bitwidth
        # self.alpha_bit = torch.nn.Parameter(torch.zeros(1), requires_grad=True)#added, to decide bitwidth
        self.grad_factor = 1.0
        self.grad_factor_1 = 1.0
        self.observer_init = torch.tensor(1, dtype=torch.int8)

        #added self.max, self.min
        min = torch.tensor([], requires_grad=False)
        max = torch.tensor([], requires_grad=False)
        self.register_buffer('min', min)
        self.register_buffer('max', max)

    def calQnQp(self):
        if self.qtype == "quint":
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2 ** self.bits - 1
            self.Qn_1 = 0
            self.Qp_1 = 2 ** (self.bits-1) - 1
            # self.Qp_1 = 2 ** (bits+1) - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = - 2 ** (self.bits - 1)
            self.Qp = 2 ** (self.bits - 1) - 1
            self.Qn_1 = - 2 ** (self.bits-1-1)
            self.Qp_1 = 2 ** (self.bits-1-1) - 1
            # self.Qn_1 = - 2 ** (bits-1+1)
            # self.Qp_1 = 2 ** (bits-1+1) - 1
        
        #added, to maintain 1 bit's calculation runnable
        if self.bits-1 == 1:
            self.Qn = 0
            self.Qp = 2 ** self.bits - 1
            self.Qn_1 = 0
            self.Qp_1 = 2 ** (self.bits-1) - 1

    def forward(self, x):
        if not self.quant:
            return x
        #bug here，not support 1 bit which makes self.Qp equals 0 so that division error
        self.grad_factor = 1.0 / math.sqrt(x.numel() * self.Qp)
        self.grad_factor_1 = 1.0 / math.sqrt(x.numel() * self.Qp_1)
        if self.observer:
            if self.per_channel:
                x_tmp = x.detach().contiguous().view(x.size()[0], -1)
                if self.observer_init == 1:
                    self.scale.data[0] = torch.mean(torch.abs(x_tmp), dim=1) * 2 / math.sqrt(self.Qp)
                    self.scale_1.data[0] = torch.mean(torch.abs(x_tmp), dim=1) * 2 / math.sqrt(self.Qp_1)
                    self.observer_init = 0
                else:
                    self.scale.data[0] = 0.9 * self.scale.data[0] + 0.1 * torch.mean(torch.abs(x_tmp), dim=1) * 2 / (
                        math.sqrt(self.Qp))
                    self.scale_1.data[0] = 0.9 * self.scale_1.data[0] + 0.1 * torch.mean(torch.abs(x_tmp), dim=1) * 2 / (
                        math.sqrt(self.Qp_1))
            else:
                if self.observer_init == 1:
                    self.scale.data[0] = torch.mean(torch.abs(x.detach()))*2/math.sqrt(self.Qp)
                    self.scale_1.data[0] = torch.mean(torch.abs(x.detach()))*2/math.sqrt(self.Qp_1)
                    self.observer_init = 0
                else:
                    self.scale.data[0] = 0.9 * self.scale.data[0] + 0.1 * torch.mean(torch.abs(x.detach()))*2/math.sqrt(self.Qp)
                    self.scale_1.data[0] = 0.9 * self.scale_1.data[0] + 0.1 * torch.mean(torch.abs(x.detach()))*2/math.sqrt(self.Qp_1)
        if self.observer or self.learning:
            # x = FunLSQ.apply(x, self.scale, self.grad_factor, self.Qn, self.Qp, self.per_channel)
            self.calcScale(x)
            x = (1-self.alpha_bit) * FunLSQ.apply(x, self.scale_1, self.grad_factor_1, self.Qn_1, self.Qp_1) + \
            self.alpha_bit * FunLSQ.apply(x, self.scale, self.grad_factor, self.Qn, self.Qp)
            # x = (1-self.alpha_bit.data[0]) * FunLSQ.apply(x, self.scale, self.grad_factor, self.Qn, self.Qp) + \
            # self.alpha_bit.data[0] * FunLSQ.apply(x, self.scale, self.grad_factor, self.Qn_1, self.Qp_1)

        print('alpha and bit are ', self.alpha_bit[0], self.bits)#test alpha and bit
        
        #added, for bit prune
        if self.alpha_bit.data[0] < 1e-4:
            self.bits = self.bits - 1
            self.alpha_bit.data[0] = 1.0
            self.calQnQp()
        # if self.alpha_bit.data[0] >= 1.0:
        #     self.bits = self.bits + 1
        #     self.alpha_bit.data[0] = 0.0
        return x
    
    def calcScale(self, tensor):
        if self.max.nelement() == 0 or self.max.data < tensor.max().data:
            self.max.data = tensor.max().data
        self.max.clamp_(min=0)

        if self.min.nelement() == 0 or self.min.data > tensor.min().data:
            self.min.data = tensor.min().data
        self.min.clamp_(max=0)

        self.scale.data[0] = (self.max - self.min) / (self.Qp - self.Qn)
        self.scale_1.data[0] = (self.max - self.min) / (self.Qp_1 - self.Qn_1)


class BPConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 bits=8, quant_wgt=True, wgt_qtype="qint", wgt_per_channel=False, quant_act=True, act_qtype="quint",
                 quant=True, observer=False, learning=False, observer_step=1):
        super(BPConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.quant = quant
        self.quant_wgt = quant_wgt
        self.quant_act = quant_act
        self.act_quantizer = LSQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        self.weight_quantizer = LSQWeightQuantizer(bits, qtype=wgt_qtype, per_channel=wgt_per_channel,
                                                   quant=quant, observer=observer, learning=learning)
        self.step = 0
        self.observer_step = observer_step

    def set_quant_config(self):
        if self.step <= self.observer_step:
            self.act_quantizer.quant = True
            self.act_quantizer.observer = True
            self.act_quantizer.learning = False

            self.weight_quantizer.quant = True
            self.weight_quantizer.observer = True
            self.weight_quantizer.learning = False
        else:
            self.act_quantizer.quant = True
            self.act_quantizer.observer = False
            self.act_quantizer.learning = True

            self.weight_quantizer.quant = True
            self.weight_quantizer.observer = False
            self.weight_quantizer.learning = True

    def forward(self, x):
        self.step += 1
        self.set_quant_config()
        if self.quant and self.quant_act:
            act = self.act_quantizer(x)
        else:
            act = x
        if self.quant and self.quant_wgt:
            wgt = self.weight_quantizer(self.weight)
        else:
            wgt = self.weight

        output = F.conv2d(act, wgt, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output
