#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
import math

__all__ = ['QILConv2d', "Transformer", "Round"]

class QILConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 bits=8, quant_wgt=True, wgt_qtype="qint", quant_act=True, act_qtype="quint",
                 quant=True, observer_step=1):
        super(QILConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.quant = quant
        self.quant_wgt = quant_wgt
        self.quant_act = quant_act
        
        self.step = 0
        self.observer_step = observer_step
        
        
        self.register_buffer('init', torch.tensor(1).float().cuda())
        # test 新增缩放因子scale 和 创建c_W, d_W, c_X, d_X
        if self.quant_wgt:
            self.c_W = nn.Parameter(data=torch.tensor(2**31 - 1).float().cuda())
            self.d_W = nn.Parameter(data=torch.tensor(2**31 - 1).float().cuda())
            # self.scale = nn.Parameter(data=torch.tensor(0.2).float().cuda())# scale，还原conv的数值范围
        if self.quant_act:
            self.c_X = nn.Parameter(data=torch.tensor(2**31 - 1).float().cuda())
            self.d_X = nn.Parameter(data=torch.tensor(2**31 - 1).float().cuda())
        
        self.wgt_qtype = wgt_qtype
        self.act_qtype = act_qtype
        self.bits = bits
        
        # self.act_quantizer = QILActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        # self.weight_quantizer = QILWeightQuantizer(bits, qtype=wgt_qtype, per_channel=wgt_per_channel,
        #                                            quant=quant, observer=observer, learning=learning)

    def forward(self, x):

        curr_running_cW = self.c_W
        curr_running_dW = self.d_W

        curr_running_cX = self.c_X
        curr_running_dX = self.d_X
        
        if self.quant and self.quant_act:
            act = self.val_quant(x, curr_running_cW, curr_running_dW, True, 'activation', self.act_qtype)
        else:
            act = x
        if self.quant and self.quant_wgt:
            wgt = self.val_quant(self.weight, curr_running_cX, curr_running_dX, False, 'weight', self.wgt_qtype)
        else:
            wgt = self.weight

        # if self.init == 1:
        #     # scale factor initialization
        #     q_output = F.conv2d(act, wgt, self.bias,  self.stride, self.padding, self.dilation, self.groups)
        #     ori_output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        #     if torch.mean(torch.abs(q_output)).data < 1e-4:
        #         self.scale.data = torch.mean(torch.abs(ori_output)) / 0.1
        #     else:
        #         self.scale.data = torch.mean(torch.abs(ori_output)) / torch.mean(torch.abs(q_output)) # beta就是s3?
        #     # if self.scale.data < 1e-4:
        #     #     self.scale.data = torch.FloatTensor(1).uniform_(0.5, 1.0).cuda() 
        #     self.init = torch.tensor(0) 
        
        output = F.conv2d(act, wgt, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # output = torch.abs(self.scale) * output
        return output
    
    def val_quant(self, x, c_delta, d_delta, gamma_fix, name, qtype):
        if qtype == "quint":
            quant_level = 2 ** self.bits - 1
        else:
            quant_level = 2 ** (self.bits - 1) - 1
        if not gamma_fix:
            gamma = torch.tensor([1.0]).cuda()
        else:
            gamma = None
            
        transform_x = self.transformer(x, c_delta, d_delta, gamma, name)
        x = Round.apply(transform_x * quant_level) / quant_level
        return x

    def transformer(self, x, c_delta, d_delta, gamma, name):
        # 疑问1，不确定是否要使用可微版本的abs
        # 疑问2，是否要保证【c和d都是正数且c>d】？
        c_delta.data = absol.apply(c_delta)
        d_delta.data = absol.apply(d_delta)

        # 防止被零除. (出现nan, inf)
        if d_delta.data < 0.001:
            d_delta.data += 1e-4

        if c_delta.data < 0.001:
            c_delta.data += 1e-4

        # 确保d_delta <= c_delta
        # bug可能存在，处理方式很重要，可能因此找不到最优
        if d_delta.data > c_delta.data:
            d_delta.data = c_delta.data

        # test NaN, 令Transformer失效，发现不报NaN的warning了，说明FunTSF我写的有问题
        x = FunTSF.apply(x, c_delta, d_delta, gamma, name)
        # print('c_delta, d_delta is respectively: \n', self.c_delta, self.d_delta, '\n')
        return x

class FunTSF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c_delta, d_delta, gamma, name):
        alpha = 0.5 / d_delta
        beta = ((- 0.5 * c_delta) / d_delta) + 0.5

        prun_point = c_delta - d_delta
        clip_point = c_delta + d_delta

        smaller = (x < prun_point).float()
        bigger = (x > clip_point).float()
        between = 1.0 - smaller - bigger  # 得到位于量化区间的index

        
        if name == 'weight':
            x = (torch.pow((alpha * torch.abs(x)) + beta, gamma) * torch.sign(x)) * between
            x = x + (torch.sign(x) * bigger)
            ctx.save_for_backward(x, between, alpha, beta, c_delta, d_delta, gamma, torch.Tensor([True]))
            return x
            
        elif name == 'activation':
            x = (alpha * x + beta) * between
            x = x + bigger
            ctx.save_for_backward(x, between, alpha, beta, c_delta, d_delta, gamma, torch.Tensor([False]))
            return x
        else:
            raise NotImplementedError
        
        return hat_x

    @staticmethod
    def backward(ctx, grad_outputs):
        x, between, alpha, beta, c_delta, d_delta, gamma, flag = ctx.saved_tensors
        grad_input = grad_c_delta = grad_d_delta = grad_gamma = name = None

        # 当输入是weight
        if flag == 1:
            # 邦哥建议： 防止零点处的梯度消失torch.sign(x)+1e-6
            common = (gamma * (alpha * absol.apply(x) + beta) ** (gamma-1))* (torch.sign(x)+1e-6) # bug可能，因为torch.sign处理求导机制不明
            grad_input = (common * (torch.sign(x)+1e-6) * alpha) * grad_outputs * between # weight 偏导数
            grad_c_delta = (-alpha * common) * grad_outputs * between # c_delta 偏导数
            grad_d_delta = (common * (alpha / d_delta) * (c_delta - absol.apply(x))) * grad_outputs * between # d_delta 偏导数
            grad_gamma = ((alpha * absol.apply(x) + beta) ** (gamma)) * \
                torch.log(alpha * absol.apply(x) + beta) * torch.sign(x) # bug可能存在，韩国人没写γ的梯度

            # test 统计张量中的0和1的个数，0多1少，bug可能，导致梯度消失
            # count_zeros = torch.sum(between == 0).item()
            # count_ones = torch.sum(between == 1).item()
            # print("Number of zeros:", count_zeros)
            # print("Number of ones:", count_ones)
            
            # print('\n---\nwgt\ngrad_output', grad_outputs)
            
            # print('\n---\nwgt\ngrad_input', grad_input)
            # print('grad_c_delta', grad_c_delta.sum(0))
            # print('grad_d_delta', grad_d_delta.sum(0))
            
            # 疑问，这里为什么使用.sum(0）沿着列维度求和
            return grad_input, grad_c_delta, grad_d_delta, None, None # test,暂时不返回grad_gamma
        
        #当输入是activation
        elif flag == 0:

            grad_input = alpha * grad_outputs * between # x 偏导数
            grad_c_delta = - alpha * grad_outputs * between # c_delta 偏导数
            grad_d_delta = (alpha / d_delta) * (c_delta - x) * grad_outputs * between # d_delta 偏导数

            # print('\n---\nact\ngrad_input', grad_input)
            # print('grad_c_delta', grad_c_delta.sum(0))
            # print('grad_d_delta', grad_d_delta.sum(0))
            
            return grad_input, grad_c_delta, grad_d_delta, None, None
        else:
            raise NotImplementedError()

# 暂时弃用
# bug可能存在, 这里的写法不对，因为重写的backward的grad是对forward的输入来说的，
# 所以我的输入应该是round(hat_w*quant_level), backward的grad应该是1/quant_level, 而且并不需要特地重写
class Discretizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hat_w, quant_level):
        ctx.save_for_backward(hat_w)
        """
            Paper中提到了使用DoReFaNet STE
            We use straight-through-estimator [2, 27] for the gradient of the discretizers.
        """
        """ Discretizer D_w Eq(2) """
        return torch.round(hat_w * quant_level) / quant_level

    @staticmethod
    def backward(ctx, grad_outputs):
        weight, = ctx.saved_tensors
        gate = (torch.abs(weight) <= 1).float()
        # 此处为什么要使用掩码，将weight中大于1的部分清零
        # 因为weight_hat的范围就是[-1, 1]
        # 但是weight_hat是经过transformer得到的，必然在[-1, 1]的范围内，所以此处的清零操作是没作用的?
        grad_inputs = grad_outputs * gate
        return grad_inputs, None

# 可微分版的绝对值函数
class absol(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.abs()

    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.sign(input)
        grad_input = grad_input + 1
        grad_input = ((grad_input+1e-6)/2).round()
        grad_input = (2*grad_input) - 1
        return grad_output * grad_input

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

# 可微分版的sign函数
class sgnol(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.sign()

    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = deriv_sign(input)
        return grad_output * grad_input

def smooth_sign(x, beta=10):
    return 1 / (1 + np.exp(-beta * x))

# 定义可微分的sign函数
def deriv_sign(x, beta=10):
    sigmoid = smooth_sign(x, beta)
    return beta * sigmoid * (1 - sigmoid)
