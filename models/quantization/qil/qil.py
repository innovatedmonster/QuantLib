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
            self.scale = nn.Parameter(data=torch.tensor(0.2).float().cuda())# scale，还原conv的数值范围
        if self.quant_act:
            self.c_X = nn.Parameter(data=torch.tensor(2**31 - 1).float().cuda())
            self.d_X = nn.Parameter(data=torch.tensor(2**31 - 1).float().cuda())
        
        # 先注释掉，令act是quint量化，wgt是qint量化
        # self.wgt_qtype = wgt_qtype
        # self.act_qtype = act_qtype
        self.bits = bits
        self.bit_range = 2**self.bits - 1
        self.bit_range_half = 2**(self.bits-1) - 1
        
    def forward(self, x):
        # daq的初始化分别是[-3.0, 3.0], [0, x.std()*3], 那qil呢？
        if self.init:
            if self.quant_wgt:
                self.c_W.data = torch.tensor(1.5).cuda()
                self.d_W.data = torch.tensor(1.5).cuda()
            if self.quant_act:
                self.c_X.data = x.std() * 3 / 2
                self.d_X.data = x.std() * 3 / 2
        
        # bug可能存在，daq分别是[lw, uw], [0, ua], 那qil呢？
        self.preProcess()
        curr_running_uw = self.c_W + self.d_W
        curr_running_lw = self.c_W - self.d_W

        curr_running_ua = self.c_X + self.d_X
        curr_running_la = self.c_X - self.d_X
        
        # Weight normalization
        mean = self.weight.data.mean().cuda()
        std = self.weight.data.std().cuda()
        norm_weight = self.weight.add(-mean).div(std)
        
        # Weight quantization
        if self.quant_wgt:
            if self.training:
                weight = self.wgt_soft_quant(norm_weight, curr_running_uw, curr_running_lw, gamma=torch.tensor([1.0]).cuda())
            else:
                weight = self.wgt_quant(norm_weight, curr_running_uw, curr_running_lw, gamma=torch.tensor([1.0]).cuda())
        else:
            weight = self.weight

        # Activation quantization
        if self.quant_act:
            if self.training:
                activation = self.act_soft_quant(x, curr_running_ua, curr_running_la)
            else:
                activation = self.act_quant(x, curr_running_ua, curr_running_la)
        else:
            activation = x

        if self.init == 1:
            # scale factor initialization
            q_output = F.conv2d(activation, weight, self.bias,  self.stride, self.padding, self.dilation, self.groups)
            ori_output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            if torch.mean(torch.abs(q_output)).data < 1e-4:
                self.scale.data = torch.mean(torch.abs(ori_output)) / 0.1
            else:
                self.scale.data = torch.mean(torch.abs(ori_output)) / torch.mean(torch.abs(q_output)) # beta就是s3?
            # if self.scale.data < 1e-4:
            #     self.scale.data = torch.FloatTensor(1).uniform_(0.5, 1.0).cuda() 
            self.init = torch.tensor(0) 
        
        output = F.conv2d(activation, weight, self.bias,  self.stride, self.padding, self.dilation, self.groups)
        output = torch.abs(self.scale) * output
        return output
    
    def clipping(self, x, upper, lower):
        # smaller = (x < lower).float()
        # bigger = (x > upper).float()
        # between = 1.0 - smaller - bigger  # 得到位于量化区间的index
        # x = x * between + upper * bigger + lower * smaller
        x = x + F.relu(lower - x)
        x = x - F.relu(x - upper)
        
        return x
    
    # added, 未知影响
    def wgt_soft_quant(self, x, u, l, gamma):
        # delta = (u - l) / self.bit_range # delta就是scale
        # interval = (x - l) / delta
        interval = torch.pow((absol.apply(x) - l) / (u - l), gamma) * torch.sign(x) * self.bit_range_half
        # interval = interval + 1/2 * torch.sign(x) * (1 - torch.sign(x)) # added, to fix low bit, but bug
        interval = torch.clamp(interval+self.bit_range_half, min=0, max=self.bit_range) # fixed, plus 1
        output = 2 * Round.apply(interval) - self.bit_range
        return output / self.bit_range

    def wgt_quant(self, x, u, l, gamma):
        # For reducing inference time
        sgn_x = torch.sign(x)
        x = self.clipping(absol.apply(x), u, l)
        interval = torch.pow((absol.apply(x) - l) / (u - l), gamma) * sgn_x * self.bit_range_half
        # interval = interval + 1/2 * torch.sign(x) * (1 - torch.sign(x)) # added, to fix low bit, but bug
        x_floor = interval.floor()
        interval = interval - x_floor
        output = 2*((interval.round()+self.bit_range_half)  + x_floor) - self.bit_range # fixed, plus 1
        return output / self.bit_range

    def act_soft_quant(self, x, u, l):
        delta = (u - l) / self.bit_range
        interval = (x - l) / delta
        interval = torch.clamp(interval, min=0, max=self.bit_range)
        output = Round.apply(interval)
        return output / self.bit_range

    def act_quant(self, x, u, l):
        # For reducing inference time
        x = self.clipping(x, u, l)
        delta = (u - l) / self.bit_range
        interval = (x - l) / delta
        x_floor = interval.floor()
        interval = interval - x_floor
        output = interval.round() + x_floor
        return output / self.bit_range

    # 对w绝对化预处理, added, 未知影响
    def preProcess(self):
        # bug可能存在
        # 疑问1，不确定是否要使用可微版本的abs
        # 疑问2，是否要保证【c和d都是正数且c>d】？
        self.c_W.data = absol.apply(self.c_W.data)
        self.d_W.data = absol.apply(self.d_W.data)

        # 防止被零除. (出现nan, inf)
        if self.d_W.data < 0.001:
            self.d_W.data += 1e-4
        if self.d_X.data < 0.001:
            self.d_X.data += 1e-4

        # 确保u和l都是正数
        # bug可能存在，处理方式很重要，可能因此找不到最优
        if self.d_W.data > self.c_W.data:
            self.d_W.data = self.c_W.data
        if self.d_X.data > self.c_X.data:
            self.d_X.data = self.c_X.data


class WgtTransformer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c_delta, d_delta, gamma):
        alpha = 0.5 / d_delta
        beta = ((- 0.5 * c_delta) / d_delta) + 0.5

        prun_point = c_delta - d_delta
        clip_point = c_delta + d_delta

        smaller = (x < prun_point).float()
        bigger = (x > clip_point).float()
        between = 1.0 - smaller - bigger  # 得到位于量化区间的index
        
        x = (torch.pow((alpha * torch.abs(x)) + beta, gamma) * torch.sign(x)) * between
        x = x + (torch.sign(x) * bigger)
        ctx.save_for_backward(x, between, alpha, beta, c_delta, d_delta, gamma)
        return x

    @staticmethod
    def backward(ctx, grad_outputs):
        x, between, alpha, beta, c_delta, d_delta, gamma = ctx.saved_tensors
        grad_input = grad_c_delta = grad_d_delta = grad_gamma = name = None

        # 邦哥建议： 防止零点处的梯度消失torch.sign(x)+1e-6
        common = (gamma * (alpha * absol.apply(x) + beta) ** (gamma-1))* (torch.sign(x)) # bug可能，因为torch.sign处理求导机制不明
        grad_input = (common * (torch.sign(x)) * alpha) * grad_outputs * between # weight 偏导数
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

class ActTransformer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c_delta, d_delta):
        alpha = 0.5 / d_delta
        beta = ((- 0.5 * c_delta) / d_delta) + 0.5

        prun_point = c_delta - d_delta
        clip_point = c_delta + d_delta

        smaller = (x < prun_point).float()
        bigger = (x > clip_point).float()
        between = 1.0 - smaller - bigger  # 得到位于量化区间的index

        x = (alpha * x + beta) * between
        x = x + bigger
        ctx.save_for_backward(x, between, alpha, beta, c_delta, d_delta)
        return x

    @staticmethod
    def backward(ctx, grad_outputs):
        x, between, alpha, beta, c_delta, d_delta = ctx.saved_tensors
        grad_input = grad_c_delta = grad_d_delta = None

        grad_input = alpha * grad_outputs * between # x 偏导数
        grad_c_delta = - alpha * grad_outputs * between # c_delta 偏导数
        grad_d_delta = (alpha / d_delta) * (c_delta - x) * grad_outputs * between # d_delta 偏导数
        
        return grad_input, grad_c_delta, grad_d_delta

# 暂时弃用
# bug可能存在, 这里的写法不对，因为重写的backward的grad是对forward的输入来说的，
# 所以我的输入应该是round(hat_w*quant_level), backward的grad应该是1/quant_level, 而且并不需要特地重写
# class Discretizer(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, hat_w, quant_level):
#         ctx.save_for_backward(hat_w)
#         """
#             Paper中提到了使用DoReFaNet STE
#             We use straight-through-estimator [2, 27] for the gradient of the discretizers.
#         """
#         """ Discretizer D_w Eq(2) """
#         return torch.round(hat_w * quant_level) / quant_level

#     @staticmethod
#     def backward(ctx, grad_outputs):
#         weight, = ctx.saved_tensors
#         gate = (torch.abs(weight) <= 1).float()
#         # 此处为什么要使用掩码，将weight中大于1的部分清零
#         # 因为weight_hat的范围就是[-1, 1]
#         # 但是weight_hat是经过transformer得到的，必然在[-1, 1]的范围内，所以此处的清零操作是没作用的?
#         grad_inputs = grad_outputs * gate
#         return grad_inputs, None

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

# 可微分版的sign函数
# class sgnol(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         return input.sign()

#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         grad_input = deriv_sign(input)
#         return grad_output * grad_input

# def smooth_sign(x, beta=10):
#     return 1 / (1 + np.exp(-beta * x))

# 定义可微分的sign函数
# def deriv_sign(x, beta=10):
#     sigmoid = smooth_sign(x, beta)
#     return beta * sigmoid * (1 - sigmoid)
