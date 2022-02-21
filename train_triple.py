'''
This script handling the training process.
'''

import argparse
import math,os
import time

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import transformer.Constants as Constants
from dataset_entity import TranslationDataset, paired_collate_fn
from transformer.Models_triple_MHA_2bert import Transformer, Transformer_bert
from transformer.Optim import ScheduledOptim
from transformer.optimization import AdamW, WarmupLinearSchedule
import random
import numpy as np
import logging
import json
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
#try:
#    from apex import amp
#except ImportError:
amp = None
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def cal_performance(pred, selector_prob, gold, opt, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, selector_prob, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, selector_prob, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.nll_loss(pred, gold, ignore_index=Constants.PAD, reduction='sum')
    return loss


def train_epoch(model, training_data, optimizer, scheduler, device, bce, smoothing, opt):
    ''' Epoch operation in training phase'''

    model.train()

    n_word_total = 0
    n_word_correct = 0
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    #celoss = torch.nn.CrossEntropyLoss()

    for step, batch in enumerate(tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False, disable=opt.local_rank not in [-1, 0])):

        # prepare data
        src_seq, src_pos, src_mask, tgt_seq, tgt_pos, tgt_mask, src_triples, _, entitys, _, entitys_mask  = map(lambda x: x.to(device), batch)
        gold = tgt_seq[:, 1:]

        # forward
        pred, selector_prob = model(src_seq, src_pos, src_mask, tgt_seq, tgt_pos, src_triples, entitys, entitys_mask, False)
        # backward
        loss, n_correct = cal_performance(pred, selector_prob, gold, opt, smoothing=smoothing)

        if opt.gradient_accumulation_steps > 1:
            loss = loss / opt.gradient_accumulation_steps
        
        if opt.fp16:
            with amp.scale_loss(loss, optimizer._optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        tr_loss += loss.item()
        # word
        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

        if (step + 1) % opt.gradient_accumulation_steps == 0:
            if opt.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), opt.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
            # update parameters
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            if global_step % opt.log_step == 0 and opt.local_rank in [-1, 0]:
                logger.info('\n(in Training) train_loss: {}, lr: {}'.format((tr_loss - logging_loss)/opt.log_step, scheduler.get_lr()[0]))
                logging_loss = tr_loss

    loss_per_word = tr_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_pos, src_mask, tgt_seq, tgt_pos, tgt_mask, src_triples, _, entitys, _, entitys_mask = map(lambda x: x.to(device), batch)
            gold = tgt_seq[:, 1:]

            # forward
            pred, selector_prob = model(src_seq, src_pos, src_mask, tgt_seq, tgt_pos, src_triples, entitys, entitys_mask, False)
            loss, n_correct = cal_performance(pred, selector_prob, gold, opt, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy

def train(model, training_data, validation_data, optimizer, scheduler, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    valid_accus = []
    valid_losss = []
    bce = torch.nn.BCEWithLogitsLoss(reduction='sum')


    for epoch_i in range(opt.epoch):
        logger.info('[ Epoch{} ]'.format(epoch_i))

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data,  optimizer, scheduler, device, bce, smoothing=opt.label_smoothing, opt=opt)
        logger.info('  - (Training) train_loss: {train_loss: 8.5f}, ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(train_loss=train_loss,
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        ##############################################################################
        if opt.local_rank not in [0, -1]:
            continue

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device, opt)
        logger.info('  - (Validation) valid_loss: {valid_loss: 8.5f}, ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(valid_loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]
        valid_losss += [valid_loss]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            save_dir = opt.save_model.split("/")[0]
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    logger.info('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    #parser.add_argument('-data', required=True)
    parser.add_argument('-data', default='data/triple.pt')
    parser.add_argument('-url', default='/shard')
    parser.add_argument('-world_size', type=int, default=4)
    
    parser.add_argument('-epoch', type=int, default=2)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("-max_grad_norm", default=1.0, type=float)

    #parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=768)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=96)
    parser.add_argument('-d_v', type=int, default=96)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-warmup_proportion', type=float, default=0.1)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default="acl_log/")
    parser.add_argument('-log_step', type=int, default=50)
    parser.add_argument('-save_model', default='output/ubuntu_512_nokg_out_model')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-warmup_rate', type=float, default=0.1)
    parser.add_argument("-adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
    parser.add_argument("-weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument('-corpus_number', type=int, default=448831)
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--bert_model", default='uncased_L-12_H-768_A-12', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument('-fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                            "See details at https://nvidia.github.io/apex/amp.html")
    
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    if opt.fp16 and amp is None:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Setup distant debugging if needed
    if opt.server_ip and opt.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        logger.info("Waiting for debugger attach")
        ptvsd.enable_attach(address=(opt.server_ip, opt.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    #========= Loading Dataset =========#
    logger.info("load {}".format(opt.data))
    data = torch.load(opt.data)
    
    logger.info("local rank : {}".format(opt.local_rank))
    if opt.local_rank == -1:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            opt.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        torch.distributed.init_process_group(backend='nccl',init_method="file://" + opt.url, rank=opt.local_rank, world_size=opt.world_size)
        opt.n_gpu = 1
    opt.device = device

    src_x = data['train']['src']
    ###################################
    opt.corpus_number = len(src_x)
    logger.info(len(src_x))
    logger.info(src_x[0])
    #opt.max_word_seq_len = data['settings'].max_word_seq_len
    #opt.max_tgt_len = data['settings'].max_tgt_len
    opt.max_token_seq_len = data['settings'].max_token_seq_len
    opt.max_token_tgt_len = data['settings'].max_token_tgt_len
    opt.entity_vocab_size = len(data['dict']['entity'])
    print("entity_vocab_size: {}".format(opt.entity_vocab_size))

    data['settings'].entity_max = 500
    training_data, validation_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    #========= Preparing Model =========#
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    logger.info(opt)
    if opt.local_rank not in [-1, 0]:
        logger.info("rank:{} barrier waiting".format(opt.local_rank))
        torch.distributed.barrier()

    entity_emb = torch.load('data/entity_emb.torch')
    #entity_emb = None

    transformer = Transformer_bert(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        opt.max_token_tgt_len,
        entity_relation_emb=entity_emb,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        bert_model = opt.bert_model,
        entity_hidden_size=100,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        num_entities=opt.entity_vocab_size,
        dropout=opt.dropout)#.to(device)

    if opt.local_rank == 0:
        logger.info("rank:{} barrier waiting".format(opt.local_rank))
        torch.distributed.barrier()

    total_gpu = torch.cuda.device_count()
    num_train_steps = int(opt.corpus_number / opt.batch_size * opt.epoch)
    gpu_batch_size = opt.batch_size/total_gpu//opt.gradient_accumulation_steps

    logger.info("total bz:{}".format(opt.batch_size))
    logger.info("gradient_accumulation_steps:{}".format(opt.gradient_accumulation_steps))
    logger.info("per gpu bz:{}".format(gpu_batch_size))
    logger.info("total train_steps:{}".format(num_train_steps))

    '''different learning rate code '''
    encoder_params = list(map(id, transformer.encoder.parameters()))
    base_params = filter(lambda p: id(p) not in encoder_params,
                         transformer.parameters())
    '''
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.999), eps=1e-06,lr=opt.lr,weight_decay=0.01),
        opt.d_model, opt.warmup_proportion, num_train_steps)
    '''
    #################################################################
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in transformer.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': opt.weight_decay},
        {'params': [p for n, p in transformer.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    warmup_steps = int(num_train_steps * opt.warmup_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.lr, eps=opt.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_steps)
    
    seed_yawa= 123456
    random.seed(seed_yawa)
    np.random.seed(seed_yawa)
    torch.manual_seed(seed_yawa)
    
    if opt.n_gpu > 0:
        torch.cuda.manual_seed_all(seed_yawa)
    
    transformer.to(opt.device)

    # should be after apex fp16
    if opt.n_gpu > 1:
        transformer = torch.nn.DataParallel(transformer)
    # fp16
    if opt.fp16:
        transformer, optimizer._optimizer = amp.initialize(transformer, optimizer._optimizer, opt_level=opt.fp16_opt_level)

    if opt.local_rank != -1:
        transformer = torch.nn.parallel.DistributedDataParallel(transformer, device_ids=[opt.local_rank],
                                                          output_device=opt.local_rank,
                                                          find_unused_parameters=True)
    
    #################################################################
    train(transformer, training_data, validation_data, optimizer, scheduler,  device, opt)


def prepare_dataloaders(data, opt):
    
    # ========= Preparing DataLoader =========#
    ######################add triple data
    n_gpu = torch.cuda.device_count()
    per_batch_size = int(opt.batch_size/n_gpu//opt.gradient_accumulation_steps)
    train_dataset = TranslationDataset(
            setting=data['settings'],
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            entity_word2idx=data['dict']['entity'],
            src_insts=data['train']['src'],
            tgt_insts=data['train']['tgt'],
            src_triples=data['train']['src_triples'],
            entitys=data['train']['entitys'],
            match_triples=data['train']['match_triples']
            )
    train_sampler = RandomSampler(train_dataset) if opt.local_rank == -1 else DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        num_workers=0,
        batch_size=per_batch_size,
        collate_fn=paired_collate_fn)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            setting=data['settings'],
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            entity_word2idx=data['dict']['entity'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt'],
            src_triples=data['valid']['src_triples'],
            entitys=data['valid']['entitys'],
            match_triples=data['valid']['match_triples']
            ),
        num_workers=0,
        batch_size=per_batch_size*opt.gradient_accumulation_steps,
        collate_fn=paired_collate_fn)
    return train_loader, valid_loader


if __name__ == '__main__':
    main()
