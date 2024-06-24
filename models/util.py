#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from models.quantization.daq.daq import DAQConv2d
from models.quantization.lsq.lsq import LSQConv2d
from models.quantization.lsq.lsq import LSQLinear
from models.quantization.lsq.lsq_plus import LSQPlusConv2d
from models.quantization.lsq.lsq_plus import LSQPlusLinear
from models.quantization.qil.qil import QILConv2d
from models.quantization.qil.qil_plus import QILPlusConv2d

# from models.quantization.bit_prune.bit_prune import BPConv2d
# from models.quantization.bit_prune.bit_prune_asc import BPConv2d
from models.quantization.bit_prune.bit_prune_bGradpScale import BPConv2d

from models.quantization.msq.msq import MSQConv2dV1
from models.quantization.msq.msq import MSQLinearV1
from models.quantization.msq.msq import MSQConv2dV2
from models.quantization.msq.msq import MSQLinearV2

from models.quantization.msq.rpcaq import RPCAQConv2d
from models.quantization.msq.rpcaq import RPCAQLinear

from models.quantization.msq.adcpq import ADCPQConv2d
from models.quantization.msq.adcpq import ADCPQLinear

def get_func(func_name):
    func = globals().get(func_name)
    return func
