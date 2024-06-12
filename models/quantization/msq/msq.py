#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math

__all__ = ['MSQConv2d', 'MSQLinear', "Round", "FunLSQ"]


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

# forward 方法的第一个参数是 ctx，它是一个用于保存信息的上下文对象。forward 方法的作用是定义前向传播的计算过程，同时将需要在后向传播时使用的信息保存到 ctx 中。
class FunMSQ(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, beta, per_channel=False):
        ctx.save_for_backward(weight, alpha, beta)
        ctx.other = g, Qn, Qp, per_channel
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)             
            alpha = torch.broadcast_to(alpha, weight.size())
            w_q = Round.apply(torch.div(weight - beta, alpha)).clamp(Qn, Qp)
            w_q = w_q * alpha + beta
            w_q = torch.transpose(w_q, 0, 1)
            w_q = w_q.contiguous().view(sizes)
        else:
            w_q = Round.apply(torch.div(weight - beta, alpha)).clamp(Qn, Qp)
            w_q = w_q * alpha + beta
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha, beta = ctx.saved_tensors
        g, Qn, Qp, per_channel = ctx.other
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            q_w = (weight - beta) / alpha
            q_w = torch.transpose(q_w, 0, 1)
            q_w = q_w.contiguous().view(sizes)
        else:
            q_w = (weight - beta) / alpha
        smaller = (q_w < Qn).float()
        bigger = (q_w > Qp).float()
        between = 1.0 - smaller - bigger  # 得到位于量化区间的index
        
        grad_beta = ((smaller + bigger) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = between * grad_weight
        return grad_weight, None, None, None, None, grad_beta


class MSQActQuantizer(nn.Module):
    def __init__(self, bits, qtype="quint", quant=False, observer=False, learning=False):
        super(MSQActQuantizer, self).__init__()
        self.bits = bits
        self.qtype = qtype
        self.quant = quant
        self.observer = observer
        self.learning = learning
        assert self.bits != 1, "MSQ don't support binary quantization"
        assert self.qtype in ("qint", "quint"), "qtype just support qint or quint"
        if self.qtype == "quint":
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2 ** self.bits - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = - 2 ** (self.bits - 1)
            self.Qp = 2 ** (self.bits - 1) - 1
        self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.zero_point = torch.nn.Parameter(torch.ones(0), requires_grad=False)
        self.grad_factor = 1.0

    def forward(self, x):
        if not self.quant:
            return x
        self.grad_factor = 1.0 / math.sqrt(x.numel() * self.Qp)
        
        min_val = torch.min(x.detach())
        self.scale.data = (torch.max(x.detach()) - min_val) / (self.Qp - self.Qn)
        self.zero_point.data = min_val - self.scale.data * self.Qn
        

        if self.observer or self.learning:
            x = FunMSQ.apply(x, self.scale, self.grad_factor, self.Qn, self.Qp)
        return x


class MSQWeightQuantizer(nn.Module):
    def __init__(self, bits, qtype="qint", per_channel=False, quant=False, observer=False, learning=False):
        super(MSQWeightQuantizer, self).__init__()
        self.bits = bits
        self.qtype = qtype
        self.quant = quant
        self.observer = observer
        self.learning = learning
        assert self.bits != 1, "LSQ don't support binary quantization"
        assert self.qtype in ("qint", "quint"), "qtype just support qint or quint"
        if self.qtype == "quint":
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2 ** bits - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = - 2 ** (bits - 1)
            self.Qp = 2 ** (bits - 1) - 1
        self.per_channel = per_channel
        self.scale = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.grad_factor = 1.0
        self.observer_init = torch.tensor(1, dtype=torch.int8)

    def forward(self, x):
        if not self.quant:
            return x
        self.grad_factor = 1.0 / math.sqrt(x.numel() * self.Qp)
        if self.observer:
            if self.per_channel:
                x_tmp = x.detach().contiguous().view(x.size()[0], -1)
                if self.observer_init == 1:
                    self.scale.data[0] = torch.mean(torch.abs(x_tmp), dim=1) * 2 / math.sqrt(self.Qp)
                    self.observer_init = 0
                else:
                    self.scale.data[0] = 0.9 * self.scale.data[0] + 0.1 * torch.mean(torch.abs(x_tmp), dim=1) * 2 / (
                        math.sqrt(self.Qp))
            else:
                if self.observer_init == 1:
                    self.scale.data[0] = torch.mean(torch.abs(x.detach()))*2/math.sqrt(self.Qp)
                    self.observer_init = 0
                else:
                    self.scale.data[0] = 0.9 * self.scale.data[0] + 0.1 * torch.mean(torch.abs(x.detach()))*2/math.sqrt(self.Qp)
        if self.observer or self.learning:
            x = FunMSQ.apply(x, self.scale, self.grad_factor, self.Qn, self.Qp, self.per_channel)
        # print('after FunLSQ, LSQWeight scale shape is ', self.scale.data.shape)
        return x


class MSQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 bits=8, quant_wgt=True, wgt_qtype="qint", wgt_per_channel=False, quant_act=True, act_qtype="quint",
                 quant=True, observer=False, learning=False, observer_step=1):
        super(MSQConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.quant = quant
        self.quant_wgt = quant_wgt
        self.quant_act = quant_act
        self.act_quantizer = MSQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        self.weight_quantizer = MSQWeightQuantizer(bits, qtype=wgt_qtype, per_channel=wgt_per_channel,
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

class MSQLinear(nn.Linear):
    def __init__(self, in_features, out_features,
                 bits=8, quant_wgt=True, wgt_qtype="qint", wgt_per_channel=False, quant_act=True, act_qtype="quint",
                 quant=True, observer=False, learning=False, observer_step=1):
        super(MSQLinear, self).__init__(in_features, out_features)
        self.quant = quant
        self.quant_wgt = quant_wgt
        self.quant_act = quant_act
        self.act_quantizer = MSQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        self.weight_quantizer = MSQWeightQuantizer(bits, qtype=wgt_qtype, per_channel=wgt_per_channel,
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

        output = F.linear(act, wgt, self.bias)
        return output