#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2023 THL A29 Limited, a Tencent company.  All rights reserved. The below software in this distribution
# may have been modified by THL A29 Limited ("Tencent Modifications"). All Tencent Modifications are Copyright (C)
# THL A29 Limited.
#


import torch


def get_all_input_quantizer(model):
    # get all input quantizer
    input_quantizer = []
    for _, mod in model.named_modules():
        if hasattr(mod, '_input_quantizer') and mod._input_quantizer._amax is not None:
            input_quantizer.append(mod._input_quantizer)
    return input_quantizer


def get_all_quantizer(model, flag = 'all'):
    # get all model quantizer
    quantizers = []
    # logger.info(f'Extracting {flag} quantizers for LSQ training!')
    for _, mod in model.named_modules():
        if flag == 'input' or flag == 'all':
            if hasattr(mod, '_input_quantizer') and mod._input_quantizer._amax is not None:
                quantizers.append(mod._input_quantizer)
        if flag == 'weight' or flag == 'all':
            if hasattr(mod, '_weight_quantizer') and mod._weight_quantizer._amax is not None:
                quantizers.append(mod._weight_quantizer)

    return quantizers


def get_weight_and_input_quantizer(model, flag = 'all', layers = None):
    # get model weight and model input_quantizer
    quantizers = [[], []]
    if layers is not None:
        layer_flag = ['layer.' + str(layer) for layer in layers]

    # logger.info(f'Extracting {flag} quantizers for LSQ training!')
    for name, mod in model.named_modules():
        if layers is not None:
            layer_key = any(flag in name for flag in layer_flag)
        else:
            layer_key = False
        if flag == 'input' or flag == 'all':
            if hasattr(mod, '_input_quantizer') and hasattr(mod._input_quantizer, '_amax') and \
                    mod._input_quantizer._amax is not None:
                quantizers[1].append(mod._input_quantizer)
        if (flag == 'weight' or flag == 'all') and not layer_key:
            if hasattr(mod, '_weight_quantizer') and hasattr(mod._weight_quantizer, '_amax') and \
                    mod._weight_quantizer._amax is not None:
                quantizers[0].append(mod._weight_quantizer)
    return quantizers


def get_all_weight_and_quantizer(model):
    # get all model weight and quantizer
    weight_and_quantizer = []
    for _, mod in model.named_modules():
        if mod.__class__.__name__ == 'LinearActivation' and hasattr(mod, '_weight_quantizer'):
            if len(mod.weight.size()) != 2:
                raise ValueError('Weight must be 2 dimension!')
            weight_and_quantizer.append([mod.weight, mod._weight_quantizer])
    return weight_and_quantizer


def get_weight_by_layer(model, layers):
    # get weight by layer
    weight_and_quantizer = []
    layer_flag = ['layer.' + str(layer) for layer in layers]
    for name, mod in model.named_modules():
        layer_key = any(flag in name for flag in layer_flag)
        if mod.__class__.__name__ == 'LinearActivation' and hasattr(mod, '_weight_quantizer') and layer_key:
            if len(mod.weight.size()) != 2:
                raise ValueError('Weight must be 2 dimension!')
            weight_and_quantizer.append([mod.weight, mod._weight_quantizer])
    # logger.info('Extracting layers from {}'.format(layer_flag))
    return weight_and_quantizer


def get_weight_amax(weight, axis=None):
    # get model weight amax
    # print(weight.abs().sum())
    if axis is None:
        return torch.max(weight.abs())
    else:
        return torch.max(weight.abs(), dim=1, keepdim=True)[0]
