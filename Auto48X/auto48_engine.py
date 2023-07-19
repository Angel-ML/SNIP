#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2023 THL A29 Limited, a Tencent company.  All rights reserved. The below software in this distribution
# may have been modified by THL A29 Limited ("Tencent Modifications"). All Tencent Modifications are Copyright (C)
# THL A29 Limited.
#


from . import rancho_distill
from . import rancho_mod_utils as extractor
from .optimization import LSQAdam
import torch.distributed as dist
import deepspeed
from . import quant_utils
import torch
from apex import amp
from . import auto48_utils
import os


def disable_quant_model(model):
    quant_utils.set_quantizer_by_name(model, [''], _disabled=True)


def get_config_path(config, args):
    file_pth = os.path.split(os.path.realpath(__file__))[0]
    if config.model_fp32 is True:
        args.deepspeed_config = file_pth + '/config/deepspeed_config_fp32.json'
    elif config.bp16:
        args.deepspeed_config = file_pth + '/config/deepspeed_config_bf16.json'
    else:
        args.deepspeed_config = file_pth + '/config/deepspeed_config.json'
    return args


def check_mode(args, model):
    # args.deepspeed = True
    # If pure distillation, then disable the quantization function for the student
    if not args.qat:
        quant_utils.set_quantizer_by_name(model, [''], _disabled=True)

    # for int8 model evaluation
    if args.pure_qat_eval:
        if args.qat:
            print("[Auto48x] pure_qat_eval mode activate！")
            quant_utils.safe_configure_model(model, args)

def Auto48Init(args, model, optimizer = None, optimizer_param_group = None, teacher_model = None, lr_scheduler = None):
    # Init Auto48 module，return model, optimizer, teacher_model, engine
    if args.ft_mode is not None:
        args = quant_utils.set_args(args)
    if (not args.qat) and (not args.pure_finetune) and (not args.pure_distillation):
        raise ValueError('Users must specify one of the following mode: qat, pure_finetune or pure_distillation')
    model.depth = model.config.num_hidden_layers
    # quant_utils.configure_model(model, args)
    if args.int4_layers is not None:
        int4_layers = [int(num) for num in args.int4_layers.split('_')]
        quant_utils.set_quantizer_by_name_and_layers(model, [''], int4_layers, _num_bits=4)
        int8_layers = [i for i in range(model.depth) if i not in int4_layers]
    else:
        int8_layers = [i for i in range(model.depth)]
    quant_utils.set_wquantizer_by_name_lcalibrator(model, [''], layers=int8_layers, _axis=(0,))
    config = auto48_utils.Auto48Config(args)

    if config.teacher_fp32 is True:
        opt_level = 'O0'
    else:
        opt_level = 'O2'

    if config.model_fp32 is True:
        model_opt_level = 'O0'
    else:
        model_opt_level = 'O2'

    args = get_config_path(config, args)

    check_mode(args, model)

    # Initializing the teacher model
    if teacher_model is not None:
        teacher_model, _ = amp.initialize(teacher_model, [], opt_level=opt_level, keep_batchnorm_fp32=True)
        quant_utils.set_quantizer_by_name(teacher_model, [''], _disabled=True)

    if not args.disable_deepspeed:
        # Use deepspeed
        model, optimizer, _, _ = deepspeed.initialize(args=args, model=model, model_parameters=optimizer_param_group,
                                                      optimizer=optimizer,
                                                      lr_scheduler=lr_scheduler,)
    else:
        # Disabling the deepspeed
        print('Disabling the deepspeed')
        model_and_optimizer = amp.initialize(model, optimizer, opt_level=model_opt_level, keep_batchnorm_fp32=True)
        if isinstance(model_and_optimizer, tuple):
            model, optimizer = model_and_optimizer
        else:
            model = model_and_optimizer
        if args.ddp:
            # Use python DDP
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model)

    engine = Auto48Engine(config, optimizer, model, teacher_model, args, lr_scheduler=lr_scheduler)
    engine.int8_layers = int8_layers
    quant_utils.convert_amax_to_fp32(model)
    return model, optimizer, teacher_model, engine


class Auto48Engine():

    def __init__(self, config, optimizer, model, teacher_model, args, lr_scheduler):
        # Init Auto48Engine args and model
        self.config = config
        self.optimizer = optimizer
        self.lsq_optimizer = None
        self.model = model
        self.teacher_model = teacher_model
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.lsq_init = False
        if not self.args.qat:
            self.calib = True
            self.calib_step = self.args.calib_step
        else:
            self.calib = False
            self.calib_step = 0

    def atomic_forward(self, *inputs, **kwargs):
        # This function is for model forward
        if self.config.KD_function == 'multilayer' or self.config.KD_function == 'hidden_contrastive':
            output_hidden_states = True
        else:
            output_hidden_states = False
        if self.config.KD_function == 'att_minilm':
            student_teacher_layer = None
            self.args.teacher_layer = None
        else:
            student_teacher_layer = -1
        self.student_outputs = self.model(*inputs, **kwargs, output_hidden_states=output_hidden_states,
                                          teacher_layer=student_teacher_layer)
        if (self.teacher_model is not None) and (not self.args.pure_finetune):
            with torch.no_grad():
                # with torch.cuda.amp.autocast(enabled=(not self.args.model_fp32)):
                self.teacher_outputs = self.teacher_model(*inputs, **kwargs, output_hidden_states=output_hidden_states,
                                                          teacher_layer=self.args.teacher_layer)
        else:
            self.teacher_outputs = None

    def engine_forward(self, *inputs, **kwargs):
        # This function is for auto48 engine forward
        if self.calib_step == 0:
            if self.args.load_from_calib:
                print('Loaded model from checkpoint with quantization scale, please make sure that amax is initialized')
                quant_utils.configure_model(self.model, self.args)
                quant_utils.check_residual(self.model)
                self.calib = True
                self.calib_step = self.args.calib_step
            else:
                if self.args.buffer_path is not None:
                    try:
                        calib_dict = torch.load(
                            self.args.buffer_path + '/pytorch_model_calibrated.bin', map_location="cpu")
                    except Exception as ex:  # pylint: disable=broad-except
                        print(ex, 'This model has not been calibrated, but no worry, will begin the calibration')
                    else:
                        model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
                        model_to_load.load_state_dict(calib_dict)
                        quant_utils.configure_model(self.model, self.args)
                        quant_utils.check_residual(self.model)
                        self.calib = True
                        self.calib_step = self.args.calib_step
        # If not been calibrated, then initialize the amax by running one forward step
        if not self.calib and self.args.qat:
            if self.calib_step == 0:
                self.enable_model_calib()
            # with torch.no_grad():
            self.student_outputs = self.model(*inputs, **kwargs)
            self.teacher_outputs = None
            self.calib_step += 1
            print(f'Calibration step {self.calib_step}/{self.args.calib_step}')
            if self.calib_step == self.args.calib_step:
                # exit()
                self.disable_model_calib()
                if self.args.local_rank != -1:
                    self.sync_amax()
                # exit()
                self.calib = True
                if self.args.buffer_path is not None:
                    print('Saving the calibrated model for future use')
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    torch.save(model_to_save.state_dict(), self.args.buffer_path + '/pytorch_model_calibrated.bin')
                quant_utils.configure_model(self.model, self.args)
                quant_utils.check_residual(self.model)
                self.atomic_forward(*inputs, **kwargs)
        else:
            self.atomic_forward(*inputs, **kwargs)

        return self.student_outputs

    def sync_amax(self):
        if dist.get_world_size() == 1:
            return
        else:
            for name, mod in self.model.module.named_modules():
                if name.endswith('quantizer'):
                    if mod._amax is not None:
                        mod._amax = mod._amax.to(device = self.model.device)
                        #print(mod._amax.device)

                        dist.all_reduce(mod._amax)
                        mod._amax.mul_(1/dist.get_world_size())

    def add_knowledge_distillation(self, loss, num_input=None, T=1, is_return_hidden_loss=False):
        if loss is None:  # For no label task
            loss = torch.torch.tensor([0.0], requires_grad=True).to(self.model.device)
        if not self.calib and self.args.qat:
            loss = loss * 0
        if self.teacher_outputs is None:
            if not is_return_hidden_loss:
                return loss
            else:
                return loss, loss
        if not self.args.no_logits:
            logits_pair = [self.student_outputs['logits'], self.teacher_outputs['logits']]
            dloss = rancho_distill.get_knowledge_distillation_loss(logits_pair[0], logits_pair[1])
        else:
            dloss = 0
        #TODO fix hidden_contrastive model output bug
        if self.config.KD_function == 'hidden_contrastive':
            attention_pair = [[self.student_outputs[0]['hidden_states'], self.student_outputs[2]],
                              [self.teacher_outputs[0]['hidden_states'], self.teacher_outputs[2]]]
        else:
            attention_pair = [[self.student_outputs['qkv'], self.student_outputs['attention_score']],
                              [self.teacher_outputs['qkv'], self.teacher_outputs['attention_score']]]

        # add pooled outputs kd
        if self.args.add_pooled_outputs_kd:
            dloss += rancho_distill.get_knowledge_distillation_loss(self.student_outputs['pooled_output'],
                                                                    self.teacher_outputs['pooled_output'])

        # switch kd function
        if self.config.KD_function == 'minilm':
            att_loss = rancho_distill.minilm(attention_pair[0], attention_pair[1], num_input,
                                             teacher_layer_num=self.args.teacher_layer, T=T)
        elif self.config.KD_function == 'minilmV2':
            att_loss = rancho_distill.minilmv2(attention_pair[0], attention_pair[1], num_input,
                                               teacher_layer_num=self.args.teacher_layer, T=T)
        elif self.config.KD_function == 'att_minilm':
            att_loss = rancho_distill.attention_minilm(attention_pair[0], attention_pair[1], num_input, T=T)
        elif self.config.KD_function == 'multilayer':
            att_loss = rancho_distill.multilayer(self.student_outputs['hidden_states'],
                                                 self.teacher_outputs['hidden_states'], T=T)
        else:
            att_loss = 0
            # print('Cannot find the corresponding distillation method for attention, therefore set this term as zero')
        dloss *= self.config.distillation_loss_scale
        att_loss *= self.config.distillation_attention_scale
        if dist.get_rank() < 1:
            print(f'Loss components: orginal loss {loss}, logits loss {dloss}, attention loss {att_loss}')
        loss = loss + dloss + att_loss
        if not is_return_hidden_loss:
            return loss
        else:
            return loss, dloss

    def update_quantization_scale(self):
        # update quantization scale
        if not self.calib:
            return
        else:
            if not self.lsq_init:
                self.init_learnable_amax()
            if self.args.local_rank == -1:
                all_reduce = False
            else:
                all_reduce = True
            self.lsq_optimizer.step(all_reduce = all_reduce)
            quant_utils.update_weight_quantizer_amax(self.model, layers=self.int8_layers)
            quant_utils.check_weight_quantizer_amax(self.model, layers=self.int8_layers)
            quant_utils.check_residual(self.model)

    def enable_model_calib(self):
        # enable model calib
        self.model.eval()
        quant_utils.enable_calibration(self.model)

    def disable_model_calib(self):
        # disable model calib
        quant_utils.finish_calibration(self.model, self.args)

    def init_learnable_amax(self):
        # Initialize the learnable amax for each quantzier, and put them into LSQ optimizers
        # Prepare the parameter group for Rancho LSQ
        if not self.args.pure_distillation and (not self.args.pure_finetune) and (not self.args.do_calib):
            all_quantizers = extractor.get_weight_and_input_quantizer(self.model, flag='all')
            for q_group in all_quantizers:
                if len(q_group) >= 0:
                    for q in q_group:
                        q.switch_learn_amax()
            lsq_params = []
            if len(all_quantizers[0]) > 0:
                lsq_params.append(
                    {'params': [p._amax for p in all_quantizers[0]], 'weight_decay': 0.0, 'lr': self.config.lr_weight})
            if len(all_quantizers[1]) > 0:
                lsq_params.append(
                    {'params': [p._amax for p in all_quantizers[1]], 'weight_decay': 0.0, 'lr': self.config.lr_input})
            # LSQ is basically the original BertAdam, but I am too lazy to change this... BTW it will automatically
            # zero the grad after each optimzier step.
            if len(lsq_params) > 0:
                lsq_optimizer = LSQAdam(lsq_params)
            else:
                lsq_optimizer = None
        else:
            lsq_optimizer = None
        self.lsq_params = lsq_params
        self.lsq_optimizer = lsq_optimizer
        self.lsq_init = True
        quant_utils.fuse_residual(self.model)

    def adding_prefinetune_noise(self, alpha=0.1):
        # adding prefinetune noise
        for name, mod in self.model.named_parameters():
            self.model.state_dict()[name][:] += (torch.rand(mod.size()).to(mod.device) - 0.5) * \
                                                alpha*torch.std(mod).to(mod.device)

    def erase_classifier(self):
        for name, mod in self.model.named_parameters():
            # print(name)
            if name.find('classifier') != -1:
                print(f'Erasing the FC layer: {name}')
                self.model.state_dict()[name][:] = torch.rand(mod.size()).to(mod.device) - 0.5
                # exit()

    def add_noise(self, loss, logits, input_list, alpha):
        # add noise
        output, _, _ = self.model(*input_list, add_noise=True)
        noise_logits = output['logits']
        loss += alpha * torch.norm(logits - noise_logits)**2

    def backward(self, loss):
        # auto48 engine backward
        if not self.calib and self.args.qat:
            return loss
        if not self.args.disable_deepspeed:
            self.model.backward(loss)
        else:
            if not self.args.model_fp32:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

    def step(self):
        # auto48 engine step
        if not self.calib and self.args.qat:
            return
        if not self.args.disable_deepspeed:
            self.model.step()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        if self.args.qat:
            self.update_quantization_scale()
