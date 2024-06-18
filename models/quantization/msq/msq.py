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
import utils.util as util

__all__ = ['MSQConv2d', 'MSQLinear', "Round", "FunMSQ"]

class FunMSQ(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, beta):
        ctx.save_for_backward(weight, alpha, beta)
        ctx.other = g, Qn, Qp
        w_q = Round.apply(torch.div((weight - beta), alpha)).clamp(Qn, Qp)
        w_q = w_q * alpha + beta
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha, beta = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = (weight - beta) / alpha
        smaller = (q_w < Qn).float()
        bigger = (q_w > Qp).float()
        between = 1.0 - smaller -bigger
        grad_beta = ((smaller + bigger) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = between * grad_weight
        return grad_weight, None,  None, None, None, grad_beta

class QParam(nn.Module):
    def __init__(self, bits, qtype="quint", quant=False, observer=False, learning=False):
        super(QParam, self).__init__()
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

class MSQActQuantizer(QParam):
    def __init__(self, bits, qtype="quint", quant=False, observer=False, learning=False):
        super(MSQActQuantizer, self).__init__(bits, qtype=qtype, quant=quant, observer=observer, learning=learning)
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


class MSQWeightQuantizer(QParam):
    def __init__(self, bits, qtype="qint", per_channel=False, quant=False, observer=False, learning=False):
        super(MSQWeightQuantizer, self).__init__(bits, qtype=qtype, quant=quant, observer=observer, learning=learning)
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

#未svd分解，scale不可导，wgt无zero_point而act有。wgt可导可不导
# 当wgt不可导，退化成朴素ptq；当wgt可导，是朴素qat
# V1主要是测试基础量化功能是否正常
class MSQConv2dV1(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 bits=8, quant_wgt=True, wgt_qtype="qint", wgt_per_channel=False, quant_act=True, act_qtype="quint",
                 quant=True, observer=False, learning=False, observer_step=1):
        super(MSQConv2dV1, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
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

#未svd分解，scale不可导，wgt无zero_point而act有。wgt可导可不导
# 当wgt不可导，退化成朴素ptq；当wgt可导，是朴素qat
class MSQLinearV1(nn.Linear):
    def __init__(self, in_features, out_features,
                 bits=8, quant_wgt=True, wgt_qtype="qint", wgt_per_channel=False, quant_act=True, act_qtype="quint",
                 quant=True, observer=False, learning=False, observer_step=1):
        super(MSQLinearV1, self).__init__(in_features, out_features)
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

#不完整，缺少img2col一步！
#svd分解，scale不可导，wgt无zero_point而act有。wgt可导可不导
# 当wgt不可导，是msq_vPtq；当wgt可导，是msq_vQat
class MSQConv2dV2(nn.Linear):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 bits=8, quant_wgt=True, wgt_qtype="qint", wgt_per_channel=False, quant_act=True, act_qtype="quint",
                 quant=True, observer=False, learning=False, observer_step=1):
        super(MSQConv2dV2, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.quant = quant
        self.quant_wgt = quant_wgt
        self.quant_act = quant_act
        self.act_quantizer_u = MSQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        self.act_quantizer_s = MSQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        self.act_quantizer_v = MSQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        self.weight_quantizer_u = MSQWeightQuantizer(bits, qtype=wgt_qtype, per_channel=wgt_per_channel,
                                                   quant=quant, observer=observer, learning=learning)
        self.weight_quantizer_s = MSQWeightQuantizer(bits, qtype=wgt_qtype, per_channel=wgt_per_channel,
                                                   quant=quant, observer=observer, learning=learning)
        self.weight_quantizer_v = MSQWeightQuantizer(bits, qtype=wgt_qtype, per_channel=wgt_per_channel,
                                                   quant=quant, observer=observer, learning=learning)
        self.step = 0
        self.observer_step = observer_step

    def wgt_quant_svd(self, U, S, VT):
        U_hat = self.weight_quantizer_u(U)
        S_hat = self.weight_quantizer_s(S)
        VT_hat = self.weight_quantizer_v(VT)
        return U_hat, S_hat, VT_hat
    
    def act_quant_svd(self, U, S, VT):
        U_hat = self.act_quantizer_u(U)
        S_hat = self.act_quantizer_s(S)
        VT_hat = self.act_quantizer_v(VT)
        return U_hat, S_hat, VT_hat

    def forward(self, x):
        self.step += 1

        # 1.svd分解
        Ux, Sx, VTx = torch.linalg.svd(x, full_matrices=False)
        Uw, Sw, VTw = torch.linalg.svd(self.weight, full_matrices=False)
        Sx = torch.diag(Sx)
        Sw = torch.diag(Sw)
        # 2.缩放
        Ux, Sx, VTx = util.scaleSVD(Ux, Sx, VTx)
        Uw, Sw, VTw = util.scaleSVD(Uw, Sw, VTw)
        # 3.分别量化并合成
        if self.quant and self.quant_act:
            # act2 = self.act_quantizer(x)
            Ux_hat, Sx_hat, VTx_hat = self.act_quant_svd(Ux, Sx, VTx)
            act = torch.mm(Ux_hat, torch.mm(Sx_hat, VTx_hat))
        else:
            act = x
        if self.quant and self.quant_wgt:
            # wgt2 = self.weight_quantizer(self.weight)
            Uw_hat, Sw_hat, VTw_hat = self.wgt_quant_svd(Uw, Sw, VTw)
            # Uw_hat, Sw_hat, VTw_hat = self.act_quant_svd(Uw, Sw, VTw)#fixed, unknown
            wgt = torch.mm(Uw_hat, torch.mm(Sw_hat, VTw_hat))
        else:
            wgt = self.weight

        output = F.conv2d(act, wgt, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # output2 = F.linear(act2, wgt2, self.bias)
        # output_ori = F.linear(x, self.weight, self.bias)
        # 4.打印量化误差
        # loss_X_svd = util.cal_quant_loss(x, act)
        # print("loss_X_svd: ", loss_X_svd)
        # loss_W_svd = util.cal_quant_loss(self.weight, wgt)
        # print("loss_W_svd: ", loss_W_svd)
        # loss_svd = util.cal_quant_loss(output, output_ori)
        # print("loss_svd: ", loss_svd)

        # loss_X_ord = util.cal_quant_loss(x, act2)
        # print("loss_X_ord: ", loss_X_ord)
        # loss_W_ord = util.cal_quant_loss(self.weight, wgt2)
        # print("loss_W_ord: ", loss_W_ord)
        # loss_ord = util.cal_quant_loss(output2, output_ori)
        # print("loss_ord: ", loss_ord)
        # print('----------------------------')

        return output

#svd分解，scale不可导，wgt无zero_point而act有。wgt可导可不导
# 当wgt不可导，是msq_vPtq；当wgt可导，是msq_vQat
class MSQLinearV2(nn.Linear):
    def __init__(self, in_features, out_features,
                 bits=8, quant_wgt=True, wgt_qtype="qint", wgt_per_channel=False, quant_act=True, act_qtype="quint",
                 quant=True, observer=False, learning=False, observer_step=1):
        super(MSQLinearV2, self).__init__(in_features, out_features)
        self.quant = quant
        self.quant_wgt = quant_wgt
        self.quant_act = quant_act
        self.act_quantizer_u = MSQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        self.act_quantizer_s = MSQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        self.act_quantizer_v = MSQActQuantizer(bits, qtype=act_qtype, quant=quant, observer=observer, learning=learning)
        self.weight_quantizer_u = MSQWeightQuantizer(bits, qtype=wgt_qtype, per_channel=wgt_per_channel,
                                                   quant=quant, observer=observer, learning=learning)
        self.weight_quantizer_s = MSQWeightQuantizer(bits, qtype=wgt_qtype, per_channel=wgt_per_channel,
                                                   quant=quant, observer=observer, learning=learning)
        self.weight_quantizer_v = MSQWeightQuantizer(bits, qtype=wgt_qtype, per_channel=wgt_per_channel,
                                                   quant=quant, observer=observer, learning=learning)
        self.step = 0
        self.observer_step = observer_step

    def wgt_quant_svd(self, U, S, VT):
        U_hat = self.weight_quantizer_u(U)
        S_hat = self.weight_quantizer_s(S)
        VT_hat = self.weight_quantizer_v(VT)
        return U_hat, S_hat, VT_hat
    
    def act_quant_svd(self, U, S, VT):
        U_hat = self.act_quantizer_u(U)
        S_hat = self.act_quantizer_s(S)
        VT_hat = self.act_quantizer_v(VT)
        return U_hat, S_hat, VT_hat

    def forward(self, x):
        self.step += 1

        # 1.svd分解
        Ux, Sx, VTx = torch.linalg.svd(x, full_matrices=False)
        Uw, Sw, VTw = torch.linalg.svd(self.weight, full_matrices=False)
        Sx = torch.diag(Sx)
        Sw = torch.diag(Sw)
        # 2.缩放
        Ux, Sx, VTx = util.scaleSVD(Ux, Sx, VTx)
        Uw, Sw, VTw = util.scaleSVD(Uw, Sw, VTw)
        # 3.分别量化并合成
        if self.quant and self.quant_act:
            # act2 = self.act_quantizer(x)
            Ux_hat, Sx_hat, VTx_hat = self.act_quant_svd(Ux, Sx, VTx)
            act = torch.mm(Ux_hat, torch.mm(Sx_hat, VTx_hat))
        else:
            act = x
        if self.quant and self.quant_wgt:
            # wgt2 = self.weight_quantizer(self.weight)
            Uw_hat, Sw_hat, VTw_hat = self.wgt_quant_svd(Uw, Sw, VTw)
            # Uw_hat, Sw_hat, VTw_hat = self.act_quant_svd(Uw, Sw, VTw)#fixed, unknown
            wgt = torch.mm(Uw_hat, torch.mm(Sw_hat, VTw_hat))
        else:
            wgt = self.weight

        output = F.linear(act, wgt, self.bias)
        # output2 = F.linear(act2, wgt2, self.bias)
        # output_ori = F.linear(x, self.weight, self.bias)
        # 4.打印量化误差
        # loss_X_svd = util.cal_quant_loss(x, act)
        # print("loss_X_svd: ", loss_X_svd)
        # loss_W_svd = util.cal_quant_loss(self.weight, wgt)
        # print("loss_W_svd: ", loss_W_svd)
        # loss_svd = util.cal_quant_loss(output, output_ori)
        # print("loss_svd: ", loss_svd)

        # loss_X_ord = util.cal_quant_loss(x, act2)
        # print("loss_X_ord: ", loss_X_ord)
        # loss_W_ord = util.cal_quant_loss(self.weight, wgt2)
        # print("loss_W_ord: ", loss_W_ord)
        # loss_ord = util.cal_quant_loss(output2, output_ori)
        # print("loss_ord: ", loss_ord)
        # print('----------------------------')

        return output
    