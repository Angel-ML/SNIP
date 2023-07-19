#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# The below software in this distribution may have been modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C)
# THL A29 Limited.
#
# Copyright (c) 2019-2020 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for training models with pytorch-quantization"""


import re
import torch
import Auto48X.pytorch_quantization as quantization
import Auto48X.pytorch_quantization.nn as quant_nn
from Auto48X.pytorch_quantization.tensor_quant import QuantDescriptor
from Auto48X.pytorch_quantization import calib


# Quant args default dict
quant_default_dict = {
    'wprec': 8, 'aprec': 8, 'quant_per_tensor': False, 'quant_disable': False, 'quant_disable_keyword': "",
    'calibrator': 'max', 'percentile': None, "fuse_qkv": False, "quant_asymmetric": False
}


class Logger:

    def info(self, s):
        print("INFO:", s)

    def warn(self, s):
        print("WARN:", s)
logger = Logger()


name_width = 50  # max width of layer names
qname_width = name_width + 20  # max width of quantizer names


def add_arguments(parser):
    """Add arguments to parser for functions defined in quant_trainer."""

    group = parser.add_argument_group('quant_trainer arguments')
    group.add_argument('--wprec', type=int, default=8,
                        help='weight precision')
    group.add_argument('--aprec', type=int, default=8,
                        help='activation precision')
    group.add_argument('--quant-per-tensor', action='store_true',
                        help='per tensor weight scaling')
    group.add_argument('--quant-disable', action='store_true',
                        help='disable all quantizers')
    group.add_argument('--quant-disable-keyword', type=str, nargs='+',
                        help='disable quantizers by keyword')
    group.add_argument('--calibrator', default='max',
                       help='which quantization range calibrator to use')
    group.add_argument('--percentile', default=None, type=float,
                       help='percentile for PercentileCalibrator')
    group.add_argument('--fuse-qkv', action='store_true',
                       help='use the same scale factor for qkv')
    group.add_argument('--quant-asymmetric', action='store_true',
                        help='use an asymmetric integer range for quantization')


def set_args(args):
    # set quant default args by ft_mode
    if args.ft_mode == 1:
        args.wprec = 8
        args.aprec = 8
        args.quant_per_tensor = False
        args.quant_disable = False
        args.quant_disable_keyword = ['final_input', 'layernorm_input', 'softmax_input', 'residual_input',
                                      'local_input', 'aftergemm']
        args.fuse_qkv = False
        args.quant_asymmetric = False
    elif args.ft_mode == 2:
        args.wprec = 8
        args.aprec = 8
        args.quant_per_tensor = True
        args.quant_disable = False
        args.quant_disable_keyword = ['final_input', 'layernorm_input', 'local_input']
        args.fuse_qkv = True
        args.quant_asymmetric = False
    elif args.ft_mode == -1:
        # for demobert
        args.wprec = 8
        args.aprec = 8
        args.quant_per_tensor = True
        args.quant_disable = False
        args.quant_disable_keyword = ['layernorm_input', 'softmax_input', 'aftergemm']
        args.fuse_qkv = True
        args.quant_asymmetric = True
    elif args.ft_mode == -2:
        # for demobert
        args.wprec = 8
        args.aprec = 8
        args.quant_per_tensor = False
        args.quant_disable = False
        args.quant_disable_keyword = ['layernorm_input', 'softmax_input', 'aftergemm']
        args.fuse_qkv = True
        args.quant_asymmetric = True
    else:
        raise ValueError("wrong argument value for 'ft_mode'")
    return args


def set_default_quantizers(args):
    """Set default quantizers before creating the model."""

    if args.calibrator == 'max':
        calib_method = 'max'
    elif args.calibrator == 'percentile':
        if args.percentile is None:
            raise ValueError('Specify --percentile when using percentile calibrator')
        calib_method = 'histogram'
    elif args.calibrator == 'mse':
        calib_method = 'histogram'
    elif args.calibrator == 'entropy':
        calib_method = 'histogram'
    else:
        raise ValueError(F'Invalid calibrator {args.calibrator}')

    input_desc = QuantDescriptor(num_bits=args.aprec,
                                 calib_method=calib_method,
                                 narrow_range=not args.quant_asymmetric,
                                 )
    weight_desc = QuantDescriptor(num_bits=args.wprec,
                                  axis=(None if args.quant_per_tensor else (0,)),
                                  )
    quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
    quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)


def safe_configure_model(model, args, calib=False):
    """Function called before the training loop."""

    logger.info('Configuring Model for Quantization')
    logger.info(F'using quantization package {quantization.__file__}')

    if args.quant_disable_keyword:
        set_quantizer_by_name(model, args.quant_disable_keyword, _disabled=True)
    #convert_amax fp16 or fp32 to fp32
    convert_amax_to_fp32(model)


def get_all_quantizer(model, flag = 'all'):
    quantizers = []
    # logger.info(f'Extracting {flag} quantizers for LSQ training!')
    for name, mod in model.named_modules():
        if name.endswith('quantizer') and hasattr(mod, '_amax') and mod._amax is not None:
        # if name.endswith('quantizer'):
            # print(name)
            quantizers.append(mod)
    return quantizers


def convert_amax_to_fp32(model):
    # Convert all amax into fp32
    all_quantizers = get_all_quantizer(model=model)
    #print(all_quantizers)
    for quantizer in all_quantizers:
        quantizer.convert_amax_f32()


def configure_model(model, args, calib=False):
    """Function called before the training loop."""

    logger.info('Configuring Model for Quantization')
    logger.info(F'using quantization package {quantization.__file__}')

    # if not calib:
    # if args.quant_disable:
    #     set_quantizer_by_name(model, [''], _disabled=True)
    #
    if args.quant_disable_keyword:
        set_quantizer_by_name(model, args.quant_disable_keyword, _disabled=True)

    # if args.fuse_qkv:

    fuse_qkv(model, args)
    convert_amax_to_fp32(model)


def enable_calibration(model):
    """Enable calibration of all *_input_quantizer modules in model."""

    logger.info("Enabling Calibration")
    for name, module in model.named_modules():
        if name.endswith("_quantizer"):
            if module._calibrator is not None:
                module._calibrator.reset()
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
            logger.info(F"{name:{qname_width}}: {module}")


def finish_calibration(model, args):
    """Disable calibration and load amax for all "*_input_quantizer modules in model."""

    logger.info("Loading calibrated amax")
    for name, module in model.named_modules():
        if name.endswith("_quantizer"):
            if module._calibrator is not None:
                # print(f'name of the calibrator is {module._calibrator}', name)
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                elif args.calibrator == "percentile":
                    module.load_calib_amax("percentile", percentile=args.percentile)
                else:
                    module.load_calib_amax(args.calibrator)
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()
    if args.fuse_qkv:
        fuse_qkv(model, args)
    # model.cuda()
    # print_quant_summary(model)


def fuse3(qq, qk, qv):
    if not hasattr(qq, '_amax') or not hasattr(qk, '_amax') or not hasattr(qv, '_amax'):
        logger.warn('missing amax buffer, unable to fuse3')
        return
    with torch.no_grad():
        q = qq._amax
        k = qk._amax
        v = qv._amax
        # print(f'q size {q.shape}, k size {k.shape}, v size {v.shape}')

        if q.numel() == 1:
            amax = max(q, k, v)
        else:
            return
        qq._amax = amax
        qk._amax = amax
        qv._amax = amax


def fuse2(qq, qk):
    # print(qq,qk)
    if not hasattr(qq, '_amax') or not hasattr(qk, '_amax'):
        logger.warn('missing amax buffer, unable to fuse2')
        return
    # with torch.no_grad():
    amax = max(qk._amax, qq._amax)
    qk._amax = amax
    qq._amax = amax


def fuse_qkv(model, args):
    """Adjust quantization ranges to match an implementation where the QKV projections are implemented with
    a single GEMM.
    Force the weight and output scale factors to match by taking the max of (Q,K,V).
    """


    for name, mod in model.named_modules():
        # print('!!!' * 30)
        # if torch.distributed.get_rank() == 0:
        # print(name)
        if name.endswith('.attention'):
            # print(mod)
            # if name.endswith('.attention.self'):
                # logger.info(f'FUSE_QKV: {name:{name_width}}')
            # print(f'******************************************Mod: {name} uses 4bits,
            # {mod.self.matmul_q_input_quantizer._num_bits}')
            # if mod.self.matmul_q_input_quantizer._num_bits == 4:
            #     print(f'******************************************Mod: {name} uses 4bits, therefore fuse its qkv')
            fuse3(mod.self.matmul_q_input_quantizer, mod.self.matmul_k_input_quantizer,
                  mod.self.matmul_v_input_quantizer)
            fuse3(mod.self.query._weight_quantizer, mod.self.key._weight_quantizer, mod.self.value._weight_quantizer)
            fuse3(mod.self.query._aftergemm_quantizer, mod.self.key._aftergemm_quantizer,
                  mod.self.value._aftergemm_quantizer)
            fuse3(mod.self.query._input_quantizer, mod.self.key._input_quantizer, mod.self.value._input_quantizer)
            # print('Before is: ',mod.self.query._weight_quantizer)
            fuse2(mod.self.query._input_quantizer, mod.output.add_residual_input_quantizer)


def fuse_residual(model):
    """Adjust quantization ranges to match an implementation where the QKV projections are implemented with
    a single GEMM.
    Force the weight and output scale factors to match by taking the max of (Q,K,V).
    """


    for name, mod in model.named_modules():
        if name.endswith('.attention'):
            fuse3(mod.self.matmul_q_input_quantizer, mod.self.matmul_k_input_quantizer,
                  mod.self.matmul_v_input_quantizer)
            fuse3(mod.self.query._aftergemm_quantizer, mod.self.key._aftergemm_quantizer,
                  mod.self.value._aftergemm_quantizer)
            fuse3(mod.self.query._input_quantizer, mod.self.key._input_quantizer, mod.self.value._input_quantizer)
        for i in range(3):
            if name.endswith('layer.{}'.format(i)):
                fuse2(mod.intermediate.dense._input_quantizer, mod.output.add_residual_input_quantizer)
                fuse2(mod.attention.output.add_residual_input_quantizer, mod.attention.self.query._input_quantizer)


def check_residual(model):
    """Adjust quantization ranges to match an implementation where the QKV projections are implemented with
    a single GEMM.
    Force the weight and output scale factors to match by taking the max of (Q,K,V).
    """
    def check3(a, b, c):
        if torch.norm(a-b) + torch.norm(b-c) > 1e-6:
            raise ValueError('Fuse 3 failed')

    def check2(a, b):
        if torch.norm(a-b) > 1e-6:
            raise ValueError('Fuse 2 failed')

    for name, mod in model.named_modules():
        if name.endswith('.attention'):
            check3(mod.self.matmul_q_input_quantizer._amax, mod.self.matmul_k_input_quantizer._amax,
                   mod.self.matmul_v_input_quantizer._amax)
            check3(mod.self.query._aftergemm_quantizer._amax, mod.self.key._aftergemm_quantizer._amax,
                   mod.self.value._aftergemm_quantizer._amax)
            check3(mod.self.query._input_quantizer._amax, mod.self.key._input_quantizer._amax,
                   mod.self.value._input_quantizer._amax)
        for i in range(3):
            if name.endswith('layer.{}'.format(i)):
                check2(mod.intermediate.dense._input_quantizer._amax, mod.output.add_residual_input_quantizer._amax)
                check2(mod.attention.output.add_residual_input_quantizer._amax,
                       mod.attention.self.query._input_quantizer._amax)


def unfuse_qkv(model, layers):
    """Adjust quantization ranges to match an implementation where the QKV projections are implemented with
     a single GEMM.
    Force the weight and output scale factors to match by taking the max of (Q,K,V).
    """

    def unfuse3(qq, qk, qv):
        if not hasattr(qq, '_amax') or not hasattr(qk, '_amax') or not hasattr(qv, '_amax'):
            logger.warn('missing amax buffer, unable to fuse')
            return
        with torch.no_grad():
            shared_amax = qq._amax
            qk._amax = torch.zeros_like(shared_amax) + qk._amax
            qv._amax = torch.zeros_like(shared_amax) + qv._amax
        logger.info(f'==================Finished Unfusion')
        qq._amax.requires_grad = False
        qk._amax.requires_grad = False
        qv._amax.requires_grad = False

    layer_flag = ['layer.' + str(layer) for layer in layers]
    for name, mod in model.named_modules():
        layer_key = any(flag in name for flag in layer_flag)
        if name.endswith('.attention') and layer_key:
            logger.info(f'！！！！！！！！！！！！！！！！！！Unfuse the weight quantizers for {name}')
            unfuse3(mod.self.query._weight_quantizer, mod.self.key._weight_quantizer, mod.self.value._weight_quantizer)


def print_quant_summary(model):
    """Print summary of all quantizer modules in the model."""

    counters = {'quantizers': 0, 'enabled_quantizers': 0,
                'weights': 0, 'quant_weights': 0, 'sparse_weights': 0,
                'params': 0, 'sparse_params': 0}
    for name, mod in model.named_modules():
        if isinstance(mod, quantization.nn.TensorQuantizer):
            print(f'{name:80} {mod}')
            counters['quantizers'] += 1
            if not mod._disabled:
                counters['enabled_quantizers'] += 1

        for pname, param in mod.named_parameters():
            if '.' in pname:
                continue
            counters['params'] += param.numel()
            # fullname = f'{name}.{pname}'
            # print(f'{fullname:80} {param.numel():12}')
            weight_quantizer = getattr(mod, '_weight_quantizer', None)
            if pname == 'weight':
                counters['weights'] += param.numel()
                if weight_quantizer is not None and not weight_quantizer._disabled:
                    counters['quant_weights'] += param.numel()
                counters['sparse_weights'] += param.eq(0).sum().item()
            counters['sparse_params'] += param.eq(0).sum().item()

    def print_fraction(a, b, counters, desc):
        va = counters[a]
        vb = counters[b]
        pct = va/vb * 100 if vb != 0 else float('NaN')
        print(f'{counters[a]:12}/{vb:12} ({pct:6.2f}%) {desc}')
    print_fraction('enabled_quantizers', 'quantizers', counters, 'TensorQuantizers enabled')
    print_fraction('quant_weights', 'weights', counters, 'Quantized weights')
    print_fraction('sparse_weights', 'weights', counters, 'Zero weights')
    print_fraction('weights', 'params', counters, 'Weight parameters')
    print('\n\n')


def set_quantizer(name, mod, quantizer, k, v):
    """Set attributes for mod.quantizer."""

    quantizer_mod = getattr(mod, quantizer, None)
    if quantizer_mod is not None:
        assert hasattr(quantizer_mod, k)
        setattr(quantizer_mod, k, v)
        # logger.info(f'{quantizer_mod}.{k} = {v}')
    else:
        logger.warn(f'{name} has no {quantizer}')


def set_quantizers(name, mod, which='both', **kwargs):
    """Set quantizer attributes for mod."""

    s = f'Warning: changing {which} quantizers of {name:{qname_width}}'
    for k, v in kwargs.items():
        s += (f' {k}={v}')
        if which in ['input', 'both']:
            set_quantizer(name, mod, '_input_quantizer', k, v)
        if which in ['weight', 'both']:
            set_quantizer(name, mod, '_weight_quantizer', k, v)
    # logger.info(s)


def set_quantizer_by_name(model, names, **kwargs):
    """Set quantizer attributes for layers where name contains a substring in names."""

    for name, mod in model.named_modules():
        if hasattr(mod, '_input_quantizer') or hasattr(mod, '_weight_quantizer'):
            for n in names:
                if re.search(n, name):
                    set_quantizers(name, mod, **kwargs)
        elif name.endswith('_quantizer'):
            for n in names:
                if re.search(n, name):
                    s = f'Warning: changing {name:{name_width}}'
                    for k, v in kwargs.items():
                        s += (f' {k}={v}')
                        setattr(mod, k, v)


def set_quantizer_by_name_and_layers(model, names, layers = None, **kwargs):
    """Set quantizer attributes for layers where name contains a substring in names."""
    if layers is None:
        layer_flag = ['layer']
        return
    else:
        layer_flag = ['layer.'+str(layer) for layer in layers]
    print(layer_flag)
    for name, mod in model.named_modules():
        layer_key = any(flag in name for flag in layer_flag)
        if layer_key:
            if hasattr(mod, '_input_quantizer') or hasattr(mod, '_weight_quantizer'):
                for n in names:
                    if re.search(n, name) and layer_key:
                        print('Only changing quantizers in {}'.format(name[name.find('layer'):name.find('layer')+9]))
                        set_quantizers(name, mod, **kwargs)
            elif name.endswith('_quantizer'):
                for n in names:
                    if re.search(n, name):
                        s = f'Warning: changing {name:{name_width}}'
                        for k, v in kwargs.items():
                            s += (f' {k}={v}')
                            setattr(mod, k, v)
                        # logger.info(s)


def set_quantizer_name_layers_mha(model, names, layers = None, **kwargs):
    """Set quantizer attributes for layers where name contains a substring in names."""
    if layers is None:
        layer_flag = ['layer']
        return
    else:
        layer_flag = ['layer.'+str(layer) for layer in layers]
    mha_flag = ['matmul_q_input_quantizer', 'matmul_k_input_quantizer', 'matmul_a_input_quantizer',
                'matmul_v_input_quantizer']
    # mha_flag = ['matmul_a_input_quantizer']
    # print(layer_flag)
    # print(mha_flag)
    for name, mod in model.named_modules():
        layer_key = any(flag in name for flag in layer_flag)
        mha_key = any(flag in name for flag in mha_flag)
        # print(name+'--->',mha_key)
        if layer_key and (not mha_key):
            if False:
            # if hasattr(mod, '_input_quantizer') or hasattr(mod, '_weight_quantizer'):
                for n in names:
                    if re.search(n, name) and layer_key:
                        logger.info(f'XXXName is : {name}')
                        set_quantizers(name, mod, **kwargs)
            elif name.endswith('_quantizer'):
                for n in names:
                    if re.search(n, name):
                        s = f'HereWarning: changing {name}'
                        for k, v in kwargs.items():
                            # logger.info()
                            s += (f' {k}={v}')
                            setattr(mod, k, v)
                        logger.info(s)


def set_wquantizer_by_name_lcalibrator(model, names, layers = None, **kwargs):
    """Set quantizer attributes for layers where name contains a substring in names."""
    if layers is None:
        layer_flag = ['layer']
    else:
        layer_flag = ['layer.'+str(layer) for layer in layers]
    for name, mod in model.named_modules():
        layer_key = any(flag in name for flag in layer_flag)
        # mha_key = any(flag in name for flag in mha_flag)
        # print(name+'--->',mha_key)
        if layer_key:
            if False:
            # if hasattr(mod, '_input_quantizer') or hasattr(mod, '_weight_quantizer'):
                for n in names:
                    if re.search(n, name) and layer_key:
                        logger.info(f'XXXName is : {name}')
                        set_quantizers(name, mod, **kwargs)
            elif name.endswith('weight_quantizer'):
                for n in names:
                    if re.search(n, name):
                        s = f'HereWarning: changing {name}'
                        for k, v in kwargs.items():
                            # logger.info()
                            s += (f' {k}={v}')
                            setattr(mod._calibrator, k, v)
                            # setattr(mod, k, v)
                        logger.info(s)


def update_weight_quantizer_amax(model, layers=None):
    """Set quantizer attributes for layers where name contains a substring in names."""
    if layers is None:
        layer_flag = ['layer']
        # return
    else:
        layer_flag = ['layer.'+str(layer) for layer in layers]
    flags = ['attention.self.query', 'attention.self.key', 'attention.self.value', 'attention.output.dense',
             'intermediate.dense', 'output.dense', 'pooler.dense']
    for name, mod in model.named_modules():
        layer_key = any(flag in name for flag in layer_flag)
        flag_key = any(name.endswith(flag) for flag in flags)
        if layer_key and flag_key:
            with torch.no_grad():
                temp_amax = torch.max(mod.weight.abs(), dim = 1)[0]
                # print(f'==========amx is {mod._weight_quantizer._amax.data[:2]} and {temp_amax[:2]} of {name}')
                mod._weight_quantizer._amax = temp_amax.float()


def check_weight_quantizer_amax(model, layers=None):
    """Set quantizer attributes for layers where name contains a substring in names."""

    if layers is None:
        return
    else:
        layer_flag = ['layer.'+str(layer) for layer in layers]
    flags = ['attention.self.query', 'attention.self.key', 'attention.self.value', 'attention.output.dense',
             'intermediate.dense', 'output.dense']
    # diff = model.module.
    for name, _ in model.named_modules():
        layer_key = any(flag in name for flag in layer_flag)
        flag_key = any(name.endswith(flag) for flag in flags)
        if layer_key and flag_key:
            #temp_amax = torch.max(mod.weight.abs(),dim = 1)[0]
            pass
