#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
import math

__all__ = ['QILPlusConv2d', "Round"]

class QILPlusConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 bits=8, quant_wgt=True, quant_act=True, quant=True, wgt_sigma=1, wgt_temp=2, act_sigma=2, act_temp=2):
        super(QILPlusConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.quant = quant
        self.quant_wgt = quant_wgt
        self.quant_act = quant_act
        
        self.q_value = torch.from_numpy(np.linspace(0, 1, 2))
        self.q_value = self.q_value.reshape(len(self.q_value), 1, 1, 1, 1).float().cuda()
        self.wgt_sigma = wgt_sigma
        self.wgt_temp = wgt_temp
        self.act_sigma = act_sigma
        self.act_temp = act_temp
        
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
        if torch.sign(x) == -1: # added
            interval = interval - 1   
        interval = torch.clamp(interval+self.bit_range_half, min=0, max=self.bit_range)
        output = 2 * self.soft_argmax(interval, self.wgt_temp, self.wgt_sigma) - self.bit_range
        return output / self.bit_range

    def wgt_quant(self, x, u, l, gamma):
        # For reducing inference time
        sgn_x = torch.sign(x)
        x = self.clipping(absol.apply(x), u, l)
        interval = torch.pow((absol.apply(x) - l) / (u - l), gamma) * sgn_x * self.bit_range_half
        x_floor = interval.floor()
        interval = interval - x_floor
        output = 2*((interval.round()+self.bit_range_half)  + x_floor) - self.bit_range
        return output / self.bit_range

    def act_soft_quant(self, x, u, l):
        delta = (u - l) / self.bit_range
        interval = (x - l) / delta
        interval = torch.clamp(interval, min=0, max=self.bit_range)
        output = self.soft_argmax(interval, self.act_temp, self.act_sigma)
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

    def soft_argmax(self, x, T, sigma):
        x_floor = x.floor()
        x = x - x_floor.detach()
        m_p = torch.exp(-absol.apply(x.unsqueeze(0).repeat(len(self.q_value), 1, 1, 1, 1) - self.q_value))

        # Get the kernel value
        max_value, max_idx = m_p.max(dim=0)
        max_idx = max_idx.unsqueeze(0).float().cuda()
        k_p = torch.exp(-(torch.pow(self.q_value-max_idx, 2).float()/(sigma**2)))

        # Get the score
        score = m_p * k_p

        # Flexible temperature
        denorm = (score[0] - score[1]).abs()
        T_ori = T
        T = T / denorm
        T = T.detach()

        tmp_score = T * score

        # weighted average using the score and temperature
        prob = torch.exp(tmp_score - tmp_score.max())
        denorm2 = prob.sum(dim=0, keepdim=True)
        prob = prob / denorm2

        q_var = self.q_value.clone()
        q_var[0] = q_var[0] - (1/(torch.exp(torch.tensor(T_ori).float()) - 1))
        q_var[1] = q_var[1] + (1/(torch.exp(torch.tensor(T_ori).float()) - 1))

        output = (q_var * prob).sum(dim=0)
        output = output + x_floor

        return output

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
