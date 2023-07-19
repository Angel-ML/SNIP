#
# Copyright (C) 2023 THL A29 Limited, a Tencent company.  All rights reserved. The below software in this distribution
# may have been modified by THL A29 Limited ("Tencent Modifications"). All Tencent Modifications are Copyright (C)
# THL A29 Limited.
#


import deepspeed
from .. import quant_utils
import torch
from .. import auto48_utils
import os


def Auto48_qat_eval(args, model, optimizer = None, optimizer_param_group = None, lr_scheduler = None):
    if args.ft_mode is not None:
        args = quant_utils.set_args(args)
    if (not args.qat) and (not args.pure_finetune) and (not args.pure_distillation) and (not args.pure_qat_eval):
        raise ValueError('Users must specify one of the following mode: qat, pure_finetune or pure_distillation')
    quant_utils.configure_model(model, args)
    quant_utils.set_wquantizer_by_name_lcalibrator(model, [''], _axis=(0,))
    config = auto48_utils.Auto48Config(args)
    file_pth = os.path.split(os.path.realpath(__file__))[0]
    if config.model_fp32 is True:
        args.deepspeed_config = file_pth + '/../config/deepspeed_config_fp32.json'
    elif config.bp16:
        args.deepspeed_config = file_pth + '/../config/deepspeed_config_bf16.json'
    else:
        args.deepspeed_config = file_pth + '/../config/deepspeed_config.json'

    args.deepspeed = True

    if args.int4_layers is not None:
        partial_layers = [int(num) for num in args.layers.split('_')]
        quant_utils.set_quantizer_by_name_and_layers(model, [''], partial_layers, _num_bits=4)

    #return model

    if not optimizer_param_group:
        param_optimizer = list(model.named_parameters())
        no_decay = []
        optimizer_param_group = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.02}
        ]
    if not optimizer:
        optimizer = torch.optim.Adam(optimizer_param_group, lr=0.001, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=0, amsgrad=False)

    # for int8 model evaluation
    if args.qat:
        quant_utils.safe_configure_model(model, args)

    if not args.disable_deepspeed:
        model, optimizer, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=optimizer_param_group,
                                                    optimizer=optimizer
                                                    #lr_scheduler=lr_scheduler,
                                                    )
    else:
        print('Disabling the deepspeed, therefore the model itself would be using FP32 for training')

    return model
