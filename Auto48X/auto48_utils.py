#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2023 THL A29 Limited, a Tencent company.  All rights reserved. The below software in this distribution
# may have been modified by THL A29 Limited ("Tencent Modifications"). All Tencent Modifications are Copyright (C)
# THL A29 Limited.
#


from . import quant_utils
import json
import deepspeed
import torch
import os


# defalut_dict
defalut_dict = {'lr_input': 0.1, 'lr_weight': 0.01, 'int4_layers': None, 'ddp': False, 'no_logits': False,
                'calib_step': 2, 'qat': False, 'annealing_T': 1, 'teacher_layer': -1, 'pure_distillation': False,
                'pure_qat_eval': False, 'pure_finetune': False, 'do_calib': False, 'KD_function': 'minilm',
                'bp16': False, "load_from_calib": False, 'teacher': None, 'distillation_loss_scale': 10,
                'distillation_attention_scale': 1, 'distillation_encode_scale': 1, 'model_fp32': False,
                'teacher_fp32': False, 'disable_deepspeed': False, 'calibrator': "max", "ft_mode": -1,
                "buffer_path": None, "sparse": False, "recompute_sparse_masks": False, "progressive_sparse": False,
                "sparse_ratio": 0.5, "sparse_step_size": 0.5, "add_pooled_outputs_kd": False}

# Auto48 Config
class Auto48Config():
    def __init__(self, args):
        # qat lr input
        self.lr_input = args.lr_input
        # qat lr weight
        self.lr_weight = args.lr_weight
        # enable int4_layers
        self.int4_layers = args.int4_layers
        # annealing temperature
        self.annealing_T = args.annealing_T
        # num of teacher layer
        self.teacher_layer = args.teacher_layer
        # auto48x mode
        self.pure_distillation = args.pure_distillation
        self.pure_finetune = args.pure_finetune
        self.do_calib = args.do_calib
        self.calib_step = args.calib_step
        self.teacher = args.teacher
        # distillation args
        self.distillation_loss_scale = args.distillation_loss_scale
        self.distillation_attention_scale = args.distillation_attention_scale
        self.distillation_encode_scale = args.distillation_encode_scale
        # distillation function select
        self.KD_function = args.KD_function
        self.model_fp32 = args.model_fp32
        self.teacher_fp32 = args.teacher_fp32
        self.bp16 = args.bp16
        self.disable_deepspeed = args.disable_deepspeed
        self.ddp = args.ddp
        self.no_logits = args.no_logits
        self.sparse = args.sparse
        self.recompute_sparse_masks = args.recompute_sparse_masks
        self.progressive_sparse = args.progressive_sparse
        self.sparse_ratio = args.sparse_ratio
        self.sparse_step_size = args.sparse_step_size
        self.pure_qat_eval = args.pure_qat_eval
        self.add_pooled_outputs_kd = args.add_pooled_outputs_kd


def get_from_args(parser):
    # Auto48 arguments
    group = parser.add_argument_group('Auto48 arguments')
    group.add_argument("--do_calib",
                        action="store_true",
                        help="Whether to run calibration of quantization ranges.")
    group.add_argument('--calib_step',
                        default=2, type=int,
                        help='Number of batches for calibration. 0 will disable calibration')
    quant_utils.add_arguments(parser)

    group.add_argument("--distillation",
                        action='store_true',
                        help="Whether or not to use the techer-student model for finetuning (Knowledge distillation)")
    group.add_argument("--teacher",
                        default=None, type=str,
                        help="teacher pytorch model file for distillation")
    group.add_argument('--distillation_loss_scale',
                        type=float, default=10,
                        help="scale applied to distillation component of loss")
    group.add_argument('--distillation_attention_scale',
                        type=float, default=1.0,
                        help="scale applied to distillation component of attention scores")
    group.add_argument('--distillation_encode_scale',
                        type=float, default=1.0,
                        help="scale applied to distillation component of encoder outputs")
    group.add_argument('--lr_input',
                        type=float, default=0.1,
                        help="learning rate of the input quantizer")
    group.add_argument('--lr_weight',
                        type=float, default=0.01,
                        help="learning rate of the weight quantizer")
    group.add_argument('--int4_layers',
                        type=str, default=None,
                        help="specifying you wanna use for 4 quantization, the rest of layers are "
                             "automatically quantized into 8bits")
    group.add_argument("--annealing_T",
                        default=1,
                        type=int,
                        help="The annealing temperature for smoothing the outputs of the teacher, it will decay by 1 "
                             "every epoch")
    group.add_argument("--teacher_layer",
                        default=-1,
                        type=int,
                        help="The teacher layer used for distillation")
    group.add_argument("--auto_qat",
                        default=0,
                        type=int,
                        help="NOT implemented yet. If auto_qat is 1, then it will load the original model and run "
                             "calibration and QAT training together, this should be used for the first time of QAT")
    group.add_argument("--pure_distillation",
                        default=False,
                        action='store_true',
                        help="Running vanilla distillation for improving the accuracy of the uncompressed model")
    group.add_argument("--pure_finetune",
                        default=False,
                        action='store_true',
                        help="Disable KD,Running vanilla finetune script for the encoder")
    group.add_argument("--pure_qat_eval",
                        default=False,
                        action='store_true',
                        help="Running qat model to put evaluation result")
    group.add_argument('--KD_function',
                       type=str, default='minilm',
                       help="specifying the distillation method you wanna use")
    group.add_argument('--model_fp32',
                        default=False,
                        action='store_true',
                        help="Use Fp32 for training model")
    group.add_argument('--teacher_fp32',
                       default=False,
                       action='store_true',
                       help="Use Fp32 for the inference of the teacher model")
    group.add_argument('--bp16',
                       default=False,
                       action='store_true',
                       help="Use Bp16 for training the student model")
    group.add_argument('--qat',
                       default=False,
                       action='store_true',
                       help="Switch into the qat mode")
    group.add_argument('--disable_deepspeed',
                       default=False,
                       action='store_true',
                       help="Disabling using deepspeed")
    group.add_argument('--ddp',
                       default=False,
                       action='store_true',
                       help="Enabling the ddp mode for torch distributed training")
    group.add_argument('--no_logits',
                       default=False,
                       action='store_true',
                       help="Not using the logits for distillation")
    group.add_argument('--ft_mode',
                       default=-1,
                       help="FT mode for default quant descripter")
    parser.add_argument("--buffer_path", default=None, type=str,
                        help="Path for saving the intermediate contents for Auto48 " )
    parser.add_argument("--load_from_calib", default=False, action='store_true',
                        help="Flag that disables the auto loading of calibrated model")
    # sparsity
    parser.add_argument("--sparse",
                        action='store_true',
                        help="Whether to sparse train")
    parser.add_argument("--recompute_sparse_masks",
                        action='store_true',
                        help="Whether or not to recompute sparse masks during sparse training after every epoch")
    parser.add_argument("--progressive_sparse",
                        action='store_true',
                        help="Whether or not to sparse the model in a progressive manner.")
    parser.add_argument("--sparse_ratio", default=0.5, type=float, required=False,
                        help="the model sparse ratio")
    parser.add_argument("--sparse_step_size", default=0.25, type=float, required=False,
                        help="the sparse step every time")
    group.add_argument('--add_pooled_outputs_kd',
                       default=False,
                       action='store_true',
                       help="is add pooled outputs kd")
    parser = deepspeed.add_config_arguments(parser)
    return parser


def get_from_config(config_path):
    # Get args from config file
    jsonfile = json.load(config_path)
    for key in defalut_dict.keys():
        if key not in jsonfile:
            jsonfile[key] = defalut_dict[key]
    return jsonfile


def add_core_argument(parser):
    group = parser.add_argument_group('Auto48 core arguments')
    file_pth = os.path.split(os.path.realpath(__file__))[0]
    group.add_argument("--auto48_config",
                       default=file_pth+"/config/auto48_default.json", type=str,
                       help="Config path for setting the argument of Auto48, if this is not None, then we will not try "
                            "loading the arguments from argparser")
    parser = get_from_args(parser)

    return parser


def config_auto48(args):
    args = args_check(args)
    # Check args and config
    adic = vars(args)
    # Check whether the user has modified the args
    if args.auto48_config is not None:
        with open(args.auto48_config, 'r') as load_f:
            load_dict = json.load(load_f)
        for key in load_dict.keys():
            if key not in defalut_dict:
                continue
            if adic[key] != defalut_dict[key] and adic[key] != load_dict[key]:
                print("[Auto48][WARNING] Parameter ", key, "=", adic[key], "Use Args ,Not Config")
            else:
                setattr(args, key, load_dict[key])

    if not args.disable_deepspeed:
        deepspeed.init_distributed()
    elif args.ddp:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    quant_utils.set_default_quantizers(args)
    args = quant_utils.set_args(args)
    # quant_utils.configure_model(args)
    return args


def args_check(args):
    """ Check whether user args are correctly set
        QAT and Pure_Finetune Cannot be set at the same time
        QAT and Pure_Distillation Cannot be set at the same time
        Pure_Finetune and Pure_Distillation Cannot be set at the same time
    """
    if args.ddp:
        args.disable_deepspeed = True
        print("[Auto48] Using DDP.")
    if args.pure_qat_eval:
        args.qat = True
    assert not (args.pure_finetune & args.qat), "[Auto48] QAT and Pure_Finetune Cannot be set at the same time!"
    assert not (args.pure_distillation & args.qat), "[Auto48] QAT and Pure_Distillation Cannot be set at the "\
                                                    "same time!"
    assert not (args.pure_distillation & args.pure_finetune), \
        "[Auto48] Pure_Finetune and Pure_Distillation Cannot be set at the same time!"
    return args


class Namespace():
    """Simple object for storing attributes.
    Implements equality by attribute names and values, and provides a simple
    string representation.
    """

    def __init__(self, dict1, dict2):
        for name in dict1:
            setattr(self, name, dict1[name])
        for name in dict2:
            setattr(self, name, dict2[name])
        file_pth = os.path.split(os.path.realpath(__file__))[0]
        self.auto48_config = file_pth + "/config/auto48_default.json"


def get_empty_args():
    # return Auto48 args to user
    return Namespace(defalut_dict, quant_utils.quant_default_dict)


def set_auto48_args(dict=None):
    # set_auto48_args is for the users don't want to incorporate the ARGS of AUTO48 into the ARGS of the user script
    auto48args = get_empty_args()
    if dict is not None:
        for key in dict:
            setattr(auto48args, key, dict[key])
    auto48args = config_auto48(auto48args)
    return auto48args
