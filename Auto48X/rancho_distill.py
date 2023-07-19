#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2023 THL A29 Limited, a Tencent company.  All rights reserved. The below software in this distribution
# may have been modified by THL A29 Limited ("Tencent Modifications"). All Tencent Modifications are Copyright (C)
# THL A29 Limited.
#


import torch
import torch.nn.functional as F
import math


def rancho_KLLoss(x, y):
    # return KL loss
    eps = 1e-16
    y = (y + eps)
    loss_pointwise = y.float() * (y.float().log() - x.float())
    return torch.mean(loss_pointwise)


def minilmv2(qkv_list, teacher_qkv_list, num_input, teacher_layer_num=-1, T = 1):
    # minilmv2 algorithm
    # The input dimension of A is [bs, num_heads, seq_len, head_dim]
    qlayer, klayer, vlayer = qkv_list[0][-1]
    teacher_qlayer, teacher_klayer, teacher_vlayer = teacher_qkv_list[0][teacher_layer_num]
    # assert , 'Please make sure the attention head number is equal in the teacher and student model'
    qq_matrix = torch.matmul(qlayer, qlayer.transpose(-1, -2))
    kk_matrix = torch.matmul(klayer, klayer.transpose(-1, -2))
    vv_matrix = torch.matmul(vlayer, vlayer.transpose(-1, -2))
    if qlayer.shape[1] != teacher_qlayer.shape[1]:
        student_head_num = qlayer.shape[1]
        b, h, s, d = teacher_qlayer.shape
        new_h = int(h*d / student_head_num)
        teacher_trunc_len = new_h * student_head_num
        teacher_qlayer = teacher_qlayer.permute(0, 2, 1, 3).view(b, s, int(h*d))[:, :, 0:teacher_trunc_len].view(
            b, s, student_head_num, new_h).permute(0, 2, 1, 3)
        teacher_klayer = teacher_klayer.permute(0, 2, 1, 3).view(b, s, int(h*d))[:, :, 0:teacher_trunc_len].view(
            b, s, student_head_num, new_h).permute(0, 2, 1, 3)
        teacher_vlayer = teacher_vlayer.permute(0, 2, 1, 3).view(b, s, int(h*d))[:, :, 0:teacher_trunc_len].view(
            b, s, student_head_num, new_h).permute(0, 2, 1, 3)
    teacher_qq_matrix = torch.matmul(teacher_qlayer, teacher_qlayer.transpose(-1, -2))
    teacher_kk_matrix = torch.matmul(teacher_klayer, teacher_klayer.transpose(-1, -2))
    teacher_vv_matrix = torch.matmul(teacher_vlayer, teacher_vlayer.transpose(-1, -2))

    kl_loss = torch.nn.KLDivLoss(reduction="mean")
    total_loss = 0.0
    for i in range(len(qlayer)):
        total_loss += kl_loss(
            F.log_softmax(qq_matrix[i, :, 0:num_input[i], 0:num_input[i]].float() / math.sqrt(qlayer.shape[-1]) / T,
                          dim=-1),
            F.softmax(teacher_qq_matrix[i, :, 0:num_input[i], 0:num_input[i]].float() / math.sqrt(
                qlayer.shape[-1]) / T, dim=-1) ) / len(qlayer)
        total_loss += kl_loss(F.log_softmax(kk_matrix[i, :, 0:num_input[i], 0:num_input[i]].float() /
                                            math.sqrt(qlayer.shape[-1]) / T, dim=-1),
                              F.softmax(teacher_kk_matrix[i, :, 0:num_input[i], 0:num_input[i]].float() /
                                        math.sqrt(qlayer.shape[-1]) / T, dim=-1)) / len(qlayer)
        total_loss += kl_loss(F.log_softmax(vv_matrix[i, :, 0:num_input[i], 0:num_input[i]].float() /
                                            math.sqrt(qlayer.shape[-1]) / T, dim=-1),
                              F.softmax(teacher_vv_matrix[i, :, 0:num_input[i], 0:num_input[i]].float() /
                                        math.sqrt(qlayer.shape[-1]) / T, dim=-1)) / len(qlayer)
    return total_loss


def minilm(att_layers, teacher_att_layers, num_input=None, teacher_layer_num=-1, T = 1):
    # The input dimension of A is [bs, num_heads, seq_len, head_dim]
    qlayer_minilm, _, vlayer = att_layers[0][-1]
    # print(f'teacher layernum is {len(teacher_att_layers[0])}')
    teacher_qlayer, teacher_klayer, teacher_vlayer = teacher_att_layers[0][teacher_layer_num]
    assert qlayer_minilm.shape[1] == teacher_qlayer.shape[1], 'Please make sure the attention head number ' \
                                                       'is equal in the teacher and student model'
    vv_matrix = torch.matmul(vlayer, vlayer.transpose(-1, -2))
    teacher_vv_matrix = torch.matmul(teacher_vlayer, teacher_vlayer.transpose(-1, -2))

    kl_loss = torch.nn.KLDivLoss(reduction="mean")
    total_loss = 0.0
    if num_input is None:
        total_loss += kl_loss(F.log_softmax(vv_matrix.float() / math.sqrt(qlayer_minilm.shape[-1]) / T, dim=-1),
                              F.softmax(teacher_vv_matrix.type(vv_matrix.dtype).float() /
                                        math.sqrt(qlayer_minilm.shape[-1]) / T,
                                        dim=-1))
        att_layer_minilm = att_layers[1][-1]
        teacher_att_layer = teacher_att_layers[1][teacher_layer_num]
        total_loss += kl_loss(torch.nn.functional.log_softmax(att_layer_minilm.float() / T, dim=-1),
                              torch.nn.functional.softmax(teacher_att_layer.type(vv_matrix.dtype).float() / T, dim=-1))
    else:
        for i in range(len(qlayer_minilm)):
            total_loss += kl_loss(F.log_softmax(vv_matrix[i, :, 0:num_input[i], 0:num_input[i]].float() /
                                                math.sqrt(qlayer_minilm.shape[-1]) / T, dim=-1),
                                  F.softmax(teacher_vv_matrix[i, :, 0:num_input[i], 0:num_input[i]].
                                            type(vv_matrix.dtype).float() /
                                            math.sqrt(qlayer_minilm.shape[-1])/T, dim=-1)) / len(qlayer_minilm)
        att_layer_minilm = att_layers[1][-1]
        teacher_att_layer = teacher_att_layers[1][teacher_layer_num]
        for i in range(len(att_layer_minilm)):
            total_loss += kl_loss(torch.nn.functional.log_softmax(att_layer_minilm[i, :, :, 0:num_input[i]].
                                                                  float() / T, dim=-1),
                             torch.nn.functional.softmax(
                                 teacher_att_layer[i, :, :, 0:num_input[i]].type(vv_matrix.dtype).float()
                                 / T, dim=-1)) / len(att_layer_minilm)

    return total_loss


def hidden_contrastive(att_layers, teacher_att_layers, num_input=None, fake_heads= 12, teacher_layer_num=-1, T = 1):
    # the contrastive of hidden att layer
    hidden = att_layers[0][-1]
    assert (hidden.shape[2] % fake_heads == 0) and (teacher_qlayer.shape[2] % fake_heads == 0),\
        'Please make sure the hidden dimension is divisibleby the fake head number'
    vv_matrix = torch.matmul(vlayer, vlayer.transpose(-1, -2))
    teacher_vv_matrix = torch.matmul(teacher_vlayer, teacher_vlayer.transpose(-1, -2))

    kl_loss = torch.nn.KLDivLoss(reduction="mean")
    total_loss = 0.0

    for i in range(len(qlayer_hidden_contrastive)):
        total_loss += kl_loss(F.log_softmax(vv_matrix[i, :, 0:num_input[i], 0:num_input[i]].float() /
                                            math.sqrt(qlayer_hidden_contrastive.shape[-1]) / T, dim=-1),
                              F.softmax(teacher_vv_matrix[i, :, 0:num_input[i], 0:num_input[i]].
                                        type(vv_matrix.dtype).float() /
                                        math.sqrt(qlayer_hidden_contrastive.shape[-1]) / T,
                                        dim=-1)) / len(qlayer_hidden_contrastive)
    att_layer = att_layers[1][-1]
    teacher_att_layer = teacher_att_layers[1][teacher_layer_num]
    for i in range(len(att_layer)):
        total_loss += kl_loss(torch.nn.functional.log_softmax(att_layer[i, :, :, 0:num_input[i]].float() / T, dim=-1),
                         torch.nn.functional.softmax(teacher_att_layer[i, :, :, 0:num_input[i]].type(vv_matrix.dtype).
                                                     float() / T, dim=-1)) / len(att_layer)

    return total_loss


def func_minilm(att_layers, teacher_att_layers, num_input, T = 1):
    # the function of minilm algorithm
    # The input dimension of A is [bs, num_heads, seq_len, head_dim]
    qlayer, _, vlayer = att_layers[0]

    teacher_qlayer, _, teacher_vlayer = teacher_att_layers[0]
    assert qlayer.shape[1] == teacher_qlayer.shape[
        1], 'Please make sure the attention head number is equal in the teacher and student model'
    vv_matrix = torch.matmul(vlayer, vlayer.transpose(-1, -2))
    teacher_vv_matrix = torch.matmul(teacher_vlayer, teacher_vlayer.transpose(-1, -2))

    kl_loss_minilm = torch.nn.KLDivLoss(reduction="mean")
    total_loss = 0.0
    for i in range(len(qlayer)):
        total_loss += kl_loss_minilm(
            F.log_softmax(vv_matrix[i, :, 0:num_input[i], 0:num_input[i]].float() / math.sqrt(qlayer.shape[-1]) / T,
                          dim=-1),
            F.softmax(teacher_vv_matrix[i, :, 0:num_input[i], 0:num_input[i]].type(vv_matrix.dtype).float() / math.sqrt(
                qlayer.shape[-1]) / T, dim=-1)) / len(qlayer)

    att_layer_minilm = att_layers[1]
    teacher_att_layer = teacher_att_layers[1]
    for i in range(len(att_layer_minilm)):
        # print(num_input[i])
        # print(len(att_layer))
        # exit()
        total_loss += kl_loss_minilm(torch.nn.functional.log_softmax(
            att_layer_minilm[i, :, :, 0:num_input[i]].float() / T, dim=-1),
            torch.nn.functional.softmax(teacher_att_layer[i, :, :, 0:num_input[i]].float() / T,
                                        dim=-1)) / len(att_layer_minilm)

    return total_loss


def attention_minilm(att_layers, teacher_att_layers, num_input):
    # attention minilm algorithm
    # return minilm loss
    student_layer_num = len(att_layers)
    teacher_layer_num = len(teacher_att_layers)
    layer_scale = teacher_layer_num // student_layer_num
    total_loss = 0
    for i, _ in enumerate(att_layers):
        if i == (student_layer_num - 1):
            layer_ids = list(range(i*layer_scale, student_layer_num))
        else:
            layer_ids = list(range(i*layer_scale, (i+1)*layer_scale))
        loss_list = []
        for teacher_id in layer_ids:
            kd_loss = func_minilm([att_layers[0][i], att_layers[1][i]], [teacher_att_layers[0][teacher_id],
                                                                         teacher_att_layers[1][teacher_id]], num_input)
            loss_list.append(kd_loss)
        with torch.no_grad():
            scale = [math.exp(-loss) for loss in loss_list]
            norm = sum(scale)

        for j in range(len(loss_list)):
            total_loss += loss_list[j] * scale[j] / norm
    return total_loss


def multilayer(hidden_states, teacher_hidden_states):
    # Return all distill loss
    distill_loss = 0
    for state, _ in zip(hidden_states, teacher_hidden_states):
        distill_loss += torch.norm(state - teacher_hidden_states.type(state.dtype), 2)
    return distill_loss


def get_knowledge_distillation_loss(output_student, output_teacher, T=1):
    # Return knowledge distillation loss
    loss_kl = rancho_KLLoss(torch.nn.functional.log_softmax(output_student / T, dim=-1), torch.nn.functional.softmax(
        output_teacher / T, dim=-1))
    return loss_kl


def get_attention_scores_loss(scores, teacher_scores, num_input):
    # get model attention scores loss
    kl_loss = torch.nn.KLDivLoss(reduction="mean")
    # print(scores[-1][0].sum(dim=-1)[0,0,0] )
    # exit()
    assert abs(scores[-1][0].sum(dim=-1)[0, 0, 0] - 1) < 1e-2, 'Input of the scores should be the probability'
    loss_kl = 0.0
    for i in range(len(scores)):
        loss_kl += kl_loss(torch.log(scores[-1][0][i, :, 0:num_input[i], 0:num_input[i]]),
                           teacher_scores[-1][0][i, :, 0:num_input[i], 0:num_input[i]]) / num_input[i]
    return loss_kl
