#
# Copyright (C) 2023 THL A29 Limited, a Tencent company.  All rights reserved. The below software in this distribution
# may have been modified by THL A29 Limited ("Tencent Modifications"). All Tencent Modifications are Copyright (C)
# THL A29 Limited.
#


import time
import os
from tqdm import tqdm
import random
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import argparse
import sys
from datasets import load_metric, load_dataset
from transformers import (
    BertConfig,
    BertTokenizer,
    DataCollatorWithPadding
)
import Auto48X
# import Auto48.models.modeling_kd as modeling
import Auto48X.models.bert as atbert


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# a fake criterion because huggingface output already has loss
def criterion(input, target):
    return input["loss"]


def trainer(model, engine, optimizer, criterion, train_dataloader, args):
    model.train()
    counter = 0
    for e in range(args.train_epoch):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for batch in (epoch_iterator):
            counter += 1
            if args.local_rank == 0:
                print("[EPOCH]:", e)
            batch.to(device)
            optimizer.zero_grad()
            outputs = engine.engine_forward(**batch)
            # pruner may wrap the criterion, for example, loss = origin_loss + norm(weight),
            # so call criterion to get loss here
            loss = criterion(outputs, None)

            num_input = torch.sum(batch["input_ids"], dim=1)
            if not args.pure_finetune:
                T = 1
                loss = engine.add_knowledge_distillation(loss, num_input=num_input, T=T)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss = loss / args.gradient_accumulation_steps

            engine.backward(loss)
            #print("Train",args.local_rank,":", counter, "/", t_total)
            if counter % args.gradient_accumulation_steps == 0 or counter == len(train_dataloader):
                engine.step()
            if counter % 800 == 0:
                print('[{}]: {}'.format(time.asctime(time.localtime(time.time())), counter))
            if counter % args.save_steps == 0:
                re = evaluator(model, metric, is_regression, validate_dataloader)
                if task_name == "mnli":
                    re2 = evaluator(model, metric, is_regression, validate_dataloader2)
                    print('Step {}: {}-{}'.format(counter // args.gradient_accumulation_steps, re, re2))
                    output_dir = os.path.join(args.output_dir, 'result{}-{}_checkpoint-{}'.format(
                        re, re2, counter))
                else:
                    print('Step {}: {}'.format(counter // args.gradient_accumulation_steps, re))
                    output_dir = os.path.join(args.output_dir, 'result{}_checkpoint-{}'.format(
                        re, counter))
                print("OUTPUTDIR", args.output_dir)
                if args.local_rank in [-1, 0] and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
            #torch.distributed.barrier()


def evaluator(model, metric, is_regression, eval_dataloader):
    model.eval()
    cont = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        cont += 1
        #print("Eval:", cont, "/", len(eval_dataloader))
        batch.to(device)
        outputs = model(**batch)
        predictions = outputs["logits"].argmax(dim=-1) if not is_regression else outputs["logits"].squeeze()
        metric.add_batch(
            predictions=predictions,
            references=batch["labels"],
        )
    return metric.compute()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--train_epoch", default=10, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, required=True,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_steps", default=100, type=int)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--taskname", type=str,  required=True,)
    parser.add_argument("--just_eval", action="store_true" )


    tasknameL = [
        "cola",
        "mnli",
        "mrpc",
        "qnli",
        "qqp",
        "rte",
        "sst2",
        "stsb",
        "wnli",
    ]
    # Setup CUDA, GPU & distributed training
    parser = Auto48X.add_core_argument(parser)
    args = parser.parse_args()
    args = Auto48X.config_auto48(args)

    if args.local_rank == -1 :
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        #torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    #os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'
    task_name = args.taskname
    is_regression = None
    if task_name == "stsb":
        is_regression = True
    else:
        is_regression = False
    num_labels = 1 if is_regression else (3 if task_name == 'mnli' else 2)

    set_seed(args)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    sentence1_key, sentence2_key = task_to_keys[task_name]

    # used to preprocess the raw data
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=False, max_length=128, truncation=True)

        if "label" in examples:
            # In all cases, rename the column to labels because the model will expect that.
            result["labels"] = examples["label"]
        return result

    raw_datasets = load_dataset(
        'glue', task_name, cache_dir='./Data/glue_data')
    processed_datasets = raw_datasets.map(
        preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names)

    train_dataset = processed_datasets['train']
    validate_dataset = processed_datasets['validation_mismatched' if task_name == "mnli" else 'validation']
    if task_name == "mnli":
        validate_dataset2 = processed_datasets['validation_matched' if task_name == "mnli" else 'validation']

    data_collator = DataCollatorWithPadding(tokenizer)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_batch_size = args.per_gpu_train_batch_size
    eval_batch_size = args.per_gpu_eval_batch_size
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, collate_fn=data_collator, batch_size=train_batch_size)
    validate_dataloader = DataLoader(validate_dataset, collate_fn=data_collator, batch_size=eval_batch_size)
    if task_name == "mnli":
        validate_dataloader2 = DataLoader(validate_dataset2, collate_fn=data_collator, batch_size=eval_batch_size)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    print("#########################", t_total, args.n_gpu)

    metric = load_metric("glue", task_name, cache_dir='./Data/glue_data')

    config = BertConfig.from_pretrained(args.model_path,
                                        num_labels=num_labels)
    config.num_labels = num_labels
    model = atbert.BertForSequenceClassification(config = config)
    teacher_model = atbert.BertForSequenceClassification(config = config)

    model.to(args.device)
    teacher_model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = Adam(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    print("############LR#############", args.lr)

    model, optimizer, teacher_model, engine = Auto48X.Auto48Init(args, model, optimizer=optimizer,
                                                                 optimizer_param_group=optimizer_grouped_parameters,
                                                                 teacher_model=teacher_model, lr_scheduler=None)

    checkpoint_state_dict = torch.load(
        args.model_path + "/pytorch_model.bin",
        map_location=torch.device("cpu"))

    missing_keys, unmatched_keys = model.module.load_state_dict(checkpoint_state_dict, strict=False)
    tmissing_keys, tunmatched_keys = teacher_model.load_state_dict(checkpoint_state_dict, strict=False)
    print(f'Missing keys are {missing_keys},{tmissing_keys}')
    print('===' * 20)
    print(f'Unmatched keys are {unmatched_keys},{tunmatched_keys}')
    print("!!load_success!!")
    print("#########################", t_total, args.n_gpu)
    if args.just_eval:
        if args.local_rank in [-1, 0] :
            re = evaluator(teacher_model, metric, is_regression, validate_dataloader)
            if task_name == "mnli":
                re2 = evaluator(teacher_model, metric, is_regression, validate_dataloader2)
                print('EVAL : {}-{}'.format( re, re2))
            else:
                print('EVAL : {}'.format( re))

        exit()

    trainer(model, engine, optimizer, criterion, train_dataloader, t_total, args)
    # print('Initial: {}'.format(evaluator(model, metric, is_regression, validate_dataloader)))


    # config_list = [{'op_types': ['Linear'], 'op_partial_names': ['bert.encoder'], 'sparsity': 0.9}]
    # p_trainer = functools.partial(trainer, train_dataloader=train_dataloader)
    #
    # # make sure you have used nni.trace to wrap the optimizer class before initialize
    # traced_optimizer = nni.trace(Adam)(model.parameters(), lr=2e-5)
    # pruner = MovementPruner(model, config_list, p_trainer, traced_optimizer, criterion, training_epochs=10,
    #                         warm_up_step=12272, cool_down_beginning_step=110448)
    #
    # _, masks = pruner.compress()
    # pruner.show_pruned_weights()
    #
    # print('Final: {}'.format(evaluator(model, metric, is_regression, validate_dataloader)))
    #
    # optimizer = Adam(model.parameters(), lr=2e-5)
    # trainer(model, optimizer, criterion, train_dataloader)
    # print('After 1 epoch finetuning: {}'.format(evaluator(model, metric, is_regression, validate_dataloader)))
