#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from models.resnet_quant import resnet20
from models.resnet_quant_all import resnet20_all

def get_model(config):
    model = globals().get(config.model.name)
    if config.model.params is None:
        return model()
    elif config.model.quant_all is False:
        return model(config.model.quant_func, config.model.quant_params, **config.model.params)
    else:
        return model(config.model.conv_quant_func, config.model.fc_quant_func,
                      config.model.quant_params, **config.model.params)


if __name__ == '__main__':
    from utils.config import load
    config = load("/home/yujin/datasets/code/QuantLib/configs/daq/resnet20_daq_W1A1.yml")
    print(config)
    model = get_model(config)
    print(model)
