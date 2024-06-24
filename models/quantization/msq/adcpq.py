#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
from ..lsq.lsq import Round
from ..lsq.lsq import FunLSQ
from ..msq.msq import FunMSQ
import utils.util as util

__all__ = ['RPCAQConv2d', 'RPCAQLinear', "Round"]

class QParam(nn.Module):
    def __init__(self, bits, qtype="quint", quant=False, observer=False, learning=False):
        super(QParam, self).__init__()
        self.bits = bits
        self.qtype = qtype
        self.quant = quant
        self.observer = observer
        self.learning = learning
        assert self.bits != 1, "RPCASQ don't support binary quantization"
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
        self.grad_factor = 1.0

        min = torch.tensor([], requires_grad=False)
        max = torch.tensor([], requires_grad=False)
        self.register_buffer('min', min)
        self.register_buffer('max', max)
    
    def updateSZ(self, x):
        if self.max.nelement() == 0 or self.max.data < x.max().data:
            self.max.data = x.max().data
        
        if self.min.nelement() == 0 or self.min.data > x.min().data:
            self.min.data = x.min().data

        min_val = self.min.detach()
        self.scale.data = (self.max.detach().detach() - min_val) / (self.Qp - self.Qn)
        if hasattr(self, 'zero_point'):
            self.zero_point.data = min_val - self.scale.data * self.Qn

    def forward(self, x):
        pass

class ADCPQActQuantizer(QParam):
    def __init__(self, bits, qtype="quint", quant=False, observer=False, learning=False):
        super(ADCPQActQuantizer, self).__init__(bits, qtype=qtype, quant=quant, observer=observer, learning=learning)
        self.zero_point = torch.nn.Parameter(torch.ones(0), requires_grad=False)

    def forward(self, x):
        if not self.quant:
            return x
        self.grad_factor = 1.0 / math.sqrt(x.numel() * self.Qp)
        
        if torch.is_grad_enabled(): #训练模式下才会更新sz
            self.updateSZ(x)

        if self.observer or self.learning:
            x = FunMSQ.apply(x, self.scale, self.grad_factor, self.Qn, self.Qp, self.zero_point)
        return x


class ADCPQWeightQuantizer(QParam):
    def __init__(self, bits, qtype="qint", per_channel=False, quant=False, observer=False, learning=False):
        super(ADCPQWeightQuantizer, self).__init__(bits, qtype=qtype, quant=quant, observer=observer, learning=learning)
        self.per_channel = per_channel

    def forward(self, x):
        if not self.quant:
            return x
        self.grad_factor = 1.0 / math.sqrt(x.numel() * self.Qp)
        
        if torch.is_grad_enabled(): #训练模式下才会更新sz
            if self.per_channel:
                x_tmp = x.detach().contiguous().view(x.size()[0], -1)
                self.updateSZ(x_tmp)
            else:
                self.updateSZ(x)

        if self.observer or self.learning:
            x = FunLSQ.apply(x, self.scale, self.grad_factor, self.Qn, self.Qp, self.per_channel)
        return x

#rpca分解，scale不可导，wgt无zero_point而act有。wgt可导可不导
# 当wgt不可导，退化成朴素ptq；当wgt可导，是朴素qat
class ADCPQConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 bits=8, quant_wgt=True, wgt_qtype="qint", wgt_per_channel=False, quant_act=True, act_qtype="quint",
                 quant=True, observer=False, learning=False, observer_step=1):
        super(ADCPQConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.quant = quant
        self.quant_wgt = quant_wgt
        self.quant_act = quant_act
        self.act_quantizer_l = ADCPQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        self.act_quantizer_s = ADCPQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        self.act_quantizer_n = ADCPQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        self.weight_quantizer_l = ADCPQWeightQuantizer(bits, qtype=wgt_qtype, per_channel=wgt_per_channel,
                                                   quant=quant, observer=observer, learning=learning)
        self.weight_quantizer_s = ADCPQWeightQuantizer(bits, qtype=wgt_qtype, per_channel=wgt_per_channel,
                                                   quant=quant, observer=observer, learning=learning)
        self.weight_quantizer_n = ADCPQWeightQuantizer(bits, qtype=wgt_qtype, per_channel=wgt_per_channel,
                                                   quant=quant, observer=observer, learning=learning)
        # self.weight_quantizer_l = ADCPQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        # self.weight_quantizer_s = ADCPQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        # self.weight_quantizer_n = ADCPQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        self.step = 0
        self.observer_step = observer_step

    def act_dcp3m(self, x, run2m=False):
        # print(x.shape)#test
        X_L = x.clone()
        X_L_hat = self.act_quantizer_l(X_L)
        X_S = x - X_L_hat
        X_S_hat = self.act_quantizer_s(X_S)
        X_N = x - X_L_hat - X_S_hat
        if not torch.all(X_N == 0).item():
            X_N_hat = self.act_quantizer_n(X_N)
        if run2m:
            X_N_hat = torch.zeros_like(x)

        return X_L_hat, X_S_hat, X_N_hat
    
    def wgt_dcp3m(self, run2m=False):
        # print(x.shape)#test
        X_L = self.weight.clone()
        X_L_hat = self.weight_quantizer_l(X_L)
        X_S = self.weight - X_L_hat
        X_S_hat = self.weight_quantizer_s(X_S)
        X_N = self.weight - X_L_hat - X_S_hat
        if not torch.all(X_N == 0).item():
            X_N_hat = self.weight_quantizer_n(X_N)
        if run2m:
            X_N_hat = torch.zeros_like(self.weight)

        return X_L_hat, X_S_hat, X_N_hat

    def forward(self, x):
        self.step += 1

        X_L_hat, X_S_hat, X_N_hat = self.act_dcp3m(x, run2m=True)
        W_L_hat, W_S_hat, W_N_hat = self.wgt_dcp3m(run2m=True)
        act = X_L_hat + X_S_hat + X_N_hat
        wgt = W_L_hat + W_S_hat + W_N_hat

        output = F.conv2d(act, wgt, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output

#rpca分解，scale不可导，wgt无zero_point而act有。wgt可导可不导
# 当wgt不可导，退化成朴素ptq；当wgt可导，是朴素qat
class ADCPQLinear(nn.Linear):
    def __init__(self, in_features, out_features,
                 bits=8, quant_wgt=True, wgt_qtype="qint", wgt_per_channel=False, quant_act=True, act_qtype="quint",
                 quant=True, observer=False, learning=False, observer_step=1):
        super(ADCPQLinear, self).__init__(in_features, out_features)
        self.quant = quant
        self.quant_wgt = quant_wgt
        self.quant_act = quant_act
        self.act_quantizer_l = ADCPQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        self.act_quantizer_s = ADCPQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        self.act_quantizer_n = ADCPQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        self.weight_quantizer_l = ADCPQWeightQuantizer(bits, qtype=wgt_qtype, per_channel=wgt_per_channel,
                                                   quant=quant, observer=observer, learning=learning)
        self.weight_quantizer_s = ADCPQWeightQuantizer(bits, qtype=wgt_qtype, per_channel=wgt_per_channel,
                                                   quant=quant, observer=observer, learning=learning)
        self.weight_quantizer_n = ADCPQWeightQuantizer(bits, qtype=wgt_qtype, per_channel=wgt_per_channel,
                                                   quant=quant, observer=observer, learning=learning)
        # self.weight_quantizer_l = ADCPQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        # self.weight_quantizer_s = ADCPQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        # self.weight_quantizer_n = ADCPQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        self.step = 0
        self.observer_step = observer_step

    def act_dcp3m(self, x, run2m=False):
        # print(x.shape)#test
        X_L = x.clone()
        X_L_hat = self.act_quantizer_l(X_L)
        X_S = x - X_L_hat
        X_S_hat = self.act_quantizer_s(X_S)
        X_N = x - X_L_hat - X_S_hat
        if not torch.all(X_N == 0).item():
            X_N_hat = self.act_quantizer_n(X_N)
        if run2m:
            X_N_hat = torch.zeros_like(x)

        return X_L_hat, X_S_hat, X_N_hat
    
    def wgt_dcp3m(self, run2m=False):
        # print(x.shape)#test
        X_L = self.weight.clone()
        X_L_hat = self.weight_quantizer_l(X_L)
        X_S = self.weight - X_L_hat
        X_S_hat = self.weight_quantizer_s(X_S)
        X_N = self.weight - X_L_hat - X_S_hat
        if not torch.all(X_N == 0).item():
            X_N_hat = self.weight_quantizer_n(X_N)
        if run2m:
            X_N_hat = torch.zeros_like(self.weight)

        return X_L_hat, X_S_hat, X_N_hat

    def forward(self, x):
        self.step += 1
        
        X_L_hat, X_S_hat, X_N_hat = self.act_dcp3m(x, run2m=True)
        W_L_hat, W_S_hat, W_N_hat = self.wgt_dcp3m(run2m=True)
        act = X_L_hat + X_S_hat + X_N_hat
        wgt = W_L_hat + W_S_hat + W_N_hat

        output = F.linear(act, wgt, self.bias)
        return output