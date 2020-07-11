# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
import math
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def print_param_stats(model, f_para_out):
    f_para_out.write("\t")
    f_para_out.write(" ".join(["%-100s"] + ["%-10s"] * 6) % \
                     ("name", "val_mean", "val_min", "val_max", "grad_mean", "grad_min", "grad_max"))
    f_para_out.write("\n")
    fmt1 = " ".join(["%-100s"] + ["%10.16f"] * 6)
    fmt2 = " ".join(["%-100s"] + ["%10.16f"] * 3 + ["%-10s"] * 3)
    for name, p in model.named_parameters():
        if hasattr(p, "grad") and p.grad is not None and p.requires_grad:
            abs_data = p.data.abs().float()
            abs_grad = p.grad.abs().float()
            val_mean, val_min, val_max, grad_mean, grad_min, grad_max = \
                abs_data.mean(), abs_data.min(), abs_data.max(), \
                abs_grad.mean(), abs_grad.min(), abs_grad.max()
            fmt = fmt1
        else:
            abs_data = p.data.abs().float()
            val_mean, val_min, val_max, grad_mean, grad_min, grad_max = \
                abs_data.mean(), abs_data.min(), abs_data.max(), \
                "", "", ""
            fmt = fmt2
        f_para_out.write("\t")
        f_para_out.write(fmt % (name, val_mean, val_min, val_max, grad_mean, grad_min, grad_max))
        f_para_out.write("\n")


def checkpoint(args, global_step, iter_id, model, optimizer):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(args.output_dir, "pytorch_model.checkpoint" + str(iter_id))
    params = {
        'model_dict': model_to_save.state_dict(),
        'args': args,
        'global_step': global_step,
        'iter_id': iter_id,
        'optimizer': optimizer.state_dict(),
    }
    try:
        torch.save(params, output_model_file)
    except BaseException:
        logger.warning('WARN: Saving failed... continuing anyway.')


def checkpoint2(args, global_step, iter_id, model, optimizer):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(args.output_dir, "pytorch_only_model.checkpoint" + str(iter_id))
    params = model_to_save.state_dict(),
    try:
        torch.save(params, output_model_file)
    except BaseException:
        logger.warning('WARN: Saving failed... continuing anyway.')


def load_checkpoint(filename, device, n_gpu):
    logger.info('Loading model %s' % filename)
    saved_params = torch.load(
        filename, map_location=lambda storage, loc: storage
    )
    args = saved_params['args']
    global_step = saved_params['global_step']
    model_dict = saved_params['model_dict']
    optimizer_dict = saved_params['optimizer']
    iter_id = saved_params['iter_id']

    model = BertForPreTraining.from_pretrained(args.bert_model, state_dict=model_dict)
    return args, global_step, iter_id, model, optimizer_dict


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--checkpoint_filename',
                        default=None,
                        type=str,
                        help="checkpoint_filename")

    args = parser.parse_args()
    os.system("clear")

    # os.system('bash /data1/private/linyankai/code/run_create.sh 0 0')

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    train_data = None
    num_train_steps = None
    if args.do_train:
        # TODO
        import indexed_dataset
        from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, BatchSampler
        import iterators
        train_data = indexed_dataset.ZhxIndexedDataset(args.data_dir)
        if args.local_rank == -1:
            # train_sampler = RandomSampler(train_data)
            train_sampler = SequentialSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_sampler = BatchSampler(train_sampler, args.train_batch_size, True)

        def collate_fn(x):
            x = torch.LongTensor([xx.numpy() for xx in x])
            return x[:, :args.max_seq_length], x[:, args.max_seq_length:2 * args.max_seq_length], x[:,
                                                                                                  2 * args.max_seq_length:3 * args.max_seq_length], x[
                                                                                                                                                    :,
                                                                                                                                                    3 * args.max_seq_length:4 * args.max_seq_length], x[
                                                                                                                                                                                                      :,
                                                                                                                                                                                                      4 * args.max_seq_length:4 * args.max_seq_length + 1]

        train_iterator = iterators.EpochBatchIterator(train_data, collate_fn, train_sampler)
        num_train_steps = int(
            len(train_data) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    # Prepare model
    config_file = os.path.join(args.bert_model, "bert_config.json")
    config = BertConfig.from_json_file(config_file)
    model = BertForPreTraining(config)
    # model = BertForPreTraining.from_pretrained(args.bert_model,
    #          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))

    logger.info(config.to_json_string())
    checkpoint_file = args.checkpoint_filename
    old_iter_id = 0
    if checkpoint_file != None:
        _, global_step, old_iter_id, model, optimizer_dict = load_checkpoint(checkpoint_file, device, n_gpu)

    if args.fp16:
        model.half()
    model.to(device)

    # if True:
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    if checkpoint_file == None:
        global_step = 0

    # optimizer.load_state_dict(optimizer_dict)
    tid = 0
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        model.train()
        import datetime
        f_config_out = open(os.path.join(args.output_dir, "config.{}".format(datetime.datetime.now())), "w")
        f_config_out.write(str(args))
        f_config_out.close()

        fout = open(os.path.join(args.output_dir, "loss.txt"), 'w')

        f_para_out = open(os.path.join(args.output_dir, "para.txt"), 'w')

        for iter_id in enumerate(tqdm(range(old_iter_id, int(args.num_train_epochs)), desc="Epoch")):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            checkpoint(args, global_step, iter_id[0] + old_iter_id, model, optimizer)
            # tid = 1 - tid
            # os.system('bash /data1/private/linyankai/code/run_create.sh '+str(random.randint(0,255))+" "+str(tid+1)+" &")
            all_loss = 0
            for step, batch in enumerate(tqdm(train_iterator.next_epoch_itr(), desc="Iteration")):
                batch = tuple(t.to(device) - 1 for t in batch)

                input_ids, input_mask, segment_ids, masked_lm_labels, next_sentence_label = batch
                loss, masked_lm_loss, next_sentence_loss = model(input_ids, segment_ids, input_mask, masked_lm_labels,
                                                                 next_sentence_label)
                # loss = masked_lm_loss
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                    masked_lm_loss = masked_lm_loss.mean()
                    next_sentence_loss = next_sentence_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if not math.isnan(loss.item()):
                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()
                all_loss += loss.item()
                if (global_step % 1000 == 0):
                    f_para_out.write("global_step:" + str(global_step) + '\n')
                    f_para_out.write("loss:\t" + str(loss) + "\n")
                    '''
                    for param_group in optimizer.param_groups:
                        f_para_out.write("lr:\t"+str(param_group['lr'])+'\n')
                        break
                    param_all= list(model.named_parameters())
                    for n, p in param_optimizer:
                        f_para_out.write(str(n)+"\t"+str(p)+"\tGradient:\t"+str(p.grad.data.float()/optimizer.cur_scale)+"\n")
                    f_para_out.flush()
                    '''
                    print_param_stats(model, f_para_out)
                if (global_step % 10 == 0):
                    fout.write("iter_id:" + str(iter_id) + "\t")
                    fout.write("global_step:" + str(global_step) + '\t')
                    fout.write("lr_this_step:" + str(
                        args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)) + '\t')
                    fout.write("step:" + str(step) + '\t')
                    fout.write("loss:" + str(loss.item()) + '\t')
                    fout.write(str(masked_lm_loss.item()) + '\t' + str(next_sentence_loss.item()) + '\n')

                if global_step % 10000 == 0:
                    checkpoint(args, global_step, iter_id[0] + old_iter_id, model, optimizer)

                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir, "pytorch_model%d.bin" % global_step)
                    torch.save(model_to_save.state_dict(), output_model_file)
                # fout.flush()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            '''
            train_iterator = None
            train_sampler = None
            train_data = None
            import indexed_dataset
            from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler,BatchSampler
            import iterators
            train_data = indexed_dataset.IndexedCachedDataset('/data'+str(tid+1)+args.data_dir[6:])
            if args.local_rank == -1:
                train_sampler = RandomSampler(train_data)
            else:
                train_sampler = DistributedSampler(train_data)
            train_sampler = BatchSampler(train_sampler, args.train_batch_size, True)
            def collate_fn(x):
              x = torch.LongTensor([xx.numpy() for xx in x])
              return x[:,:args.max_seq_length], x[:,args.max_seq_length:2*args.max_seq_length], x[:,2*args.max_seq_length:3*args.max_seq_length], x[:,3*args.max_seq_length:4*args.max_seq_length], x[:,4*args.max_seq_length:4*args.max_seq_length+1]
            train_iterator = iterators.EpochBatchIterator(train_data, collate_fn, train_sampler)
            '''

        fout.close()
        f_para_out.close()
    # Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)

    # Load a trained model that you have fine-tuned
    # model_state_dict = torch.load(output_model_file)
    # model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict)
    # model.to(device)

    # if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    #     eval_examples = processor.get_dev_examples(args.data_dir)
    #     eval_features = convert_examples_to_features(
    #         eval_examples, label_list, args.max_seq_length, tokenizer)
    #     logger.info("***** Running evaluation *****")
    #     logger.info("  Num examples = %d", len(eval_examples))
    #     logger.info("  Batch size = %d", args.eval_batch_size)
    #     all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    #     all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    #     all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    #     all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    #     eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    #     # Run prediction for full data
    #     eval_sampler = SequentialSampler(eval_data)
    #     eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    #     model.eval()
    #     eval_loss, eval_accuracy = 0, 0
    #     nb_eval_steps, nb_eval_examples = 0, 0
    #     for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
    #         input_ids = input_ids.to(device)
    #         input_mask = input_mask.to(device)
    #         segment_ids = segment_ids.to(device)
    #         label_ids = label_ids.to(device)

    #         with torch.no_grad():
    #             tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
    #             logits = model(input_ids, segment_ids, input_mask)

    #         logits = logits.detach().cpu().numpy()
    #         label_ids = label_ids.to('cpu').numpy()
    #         tmp_eval_accuracy = accuracy(logits, label_ids)

    #         eval_loss += tmp_eval_loss.mean().item()
    #         eval_accuracy += tmp_eval_accuracy

    #         nb_eval_examples += input_ids.size(0)
    #         nb_eval_steps += 1

    #     eval_loss = eval_loss / nb_eval_steps
    #     eval_accuracy = eval_accuracy / nb_eval_examples

    #     result = {'eval_loss': eval_loss,
    #               'eval_accuracy': eval_accuracy,
    #               'global_step': global_step,
    #               'loss': tr_loss/nb_tr_steps}

    #     output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    #     with open(output_eval_file, "w") as writer:
    #         logger.info("***** Eval results *****")
    #         for key in sorted(result.keys()):
    #             logger.info("  %s = %s", key, str(result[key]))
    #             writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
