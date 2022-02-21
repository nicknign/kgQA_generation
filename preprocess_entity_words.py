#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:56:13 2019

@author: rain
"""

''' Handling the data io '''
import argparse
import torch
import re
import sys
import collections
import os
import json
import tokenization
import numpy as np
from tqdm import tqdm
import time

#SPE_STR = '*@#$*($#@*@#$'
SPE_STR = '+++'
# from attention_is_all_you_need.transformer import Constants as Constants
PAD = 0
UNK = 100
CLS = 101
SEP = 102

PAD_WORD = '[PAD]'
UNK_WORD = '[UNK]'
CLS_WORD = '[CLS]'
SEP_WORD = '[SEP]'

from tqdm import tqdm
#from nltk.stem import SnowballStemmer
base_path = os.path.dirname(os.path.abspath(__file__)) + '/'
#stemmer = SnowballStemmer("english")

import spacy
#nlp = spacy.load(base_path + "en_core_web_sm")
BERT_SP = u'●'
import string
PUNC = set(string.punctuation)


def checkLength(tochecklist):
    max_len = 0
    max_seq = None
    for item in tochecklist:
        try:
            if len(item) > max_len:
                max_len = len(item)
                max_seq = item
        except:
            aa =1#print(item)
    print('#########################')
    print("max_len:%d"%max_len)
    print(max_seq)
    print('################################')


# no pad in this function
def read_instances_from_data_file(opt, entity_vocab, istrain=True):
    tokenizer = tokenization.FullTokenizer(
      vocab_file=opt.bert_vocab_file, do_lower_case=opt.do_lower_case)
    with open('%s/resource.txt' % opt.data_dir) as f:
        d = json.loads(f.readline())
    
    csk_triples = d['csk_triples']
    csk_entities = d['csk_entities']
    kb_dict = d['dict_csk']
    NAF = [0, 0, 0]

    data = []
    #data_file = '{}/trainset30w.txt'.format(opt.data_dir) if istrain else '{}/validset.txt'.format(opt.data_dir)
    data_file = '{}/validset.txt'.format(opt.data_dir) if istrain else '{}/testset.txt'.format(opt.data_dir)
    with open(data_file) as f:
        for idx, line in enumerate(f):
            if idx % 100000 == 0: print('read train file line %d' % idx)
            data.append(json.loads(line))
    src_word_insts = []
    tgt_word_insts = []
    entitys = []
    split_map = []
    src_triples, tgt_triples, match_triples = [], [], []
    post_triple_index = []
    bert2post_index = []

    encoder_len = opt.max_token_seq_len
    decoder_len = opt.max_token_tgt_len

    print("generating example data...")
    for index, example in enumerate(tqdm(data)):
        #src_word
        src_line = ' '.join(example['post'])
        src_line = src_line.replace(" n't", "n't")
        src_tokens = tokenizer.tokenize(src_line)
        src_word_inst = tokenizer.convert_tokens_to_ids(src_tokens)
        src_word_inst = [CLS] + src_word_inst + [SEP]
        src_word_insts.append(src_word_inst)

        #TODO: map the bert and post
        bert2post = [0] * len(src_tokens)
        post_ptr = 0
        posts = example['post']

        for in_s, i in enumerate(src_tokens):
            bert2post[in_s] = post_ptr
            assert post_ptr < len(posts)
            if ''.join(src_tokens[:in_s+1]).replace('##', '').find(''.join(posts[:post_ptr+1])) != -1:
                post_ptr += 1

        
        assert post_ptr != len(src_tokens) + 1
        bert2post_index.append(bert2post)
        
        #tgt_word
        tgt_line = ' '.join(example['response'])
        tgt_line = tgt_line.replace(" n't", "n't")
        tgt_tokens = tokenizer.tokenize(tgt_line)
        tgt_word_inst = tokenizer.convert_tokens_to_ids(tgt_tokens)
        tgt_word_inst = [CLS] + tgt_word_inst + [SEP]
        tgt_word_insts.append(tgt_word_inst)
        
        #entitys
        # 将entity连接在一起供copy使用
        entities = [csk_entities[x] for entity in example['all_entities'] for x in entity]
        entities_line = ' '.join(entities)
        entities_line = entities_line.replace(" n't", "n't")
        entities_tokens = tokenizer.tokenize(entities_line)
        entities_inst = tokenizer.convert_tokens_to_ids(entities_tokens)
        entitys.append(entities_inst)

        triples_index = example['post_triples']
        post_triple_index.append(triples_index)

        #src_triples
        row = []
        match_inst = []
        for triple in example['all_triples']:
            triple_ids = []
            for x in triple:
                if x in example['response_triples']:
                    match_inst.append(1)
                else:
                    match_inst.append(0)
                triple_ids.append([entity_vocab.get(i, 0) for i in csk_triples[x].split(', ')])
            row.extend(triple_ids)
        src_triples.append(row)
        match_triples.append(match_inst)
        
        onelline_row = sum(row, [])
        assert len(onelline_row) == len(match_inst)

        #response_triples
        row = []
        for x in example['response_triples']:
            if x == -1:
                continue
            else:
                row.append([entity_vocab.get(i, 0) for i in csk_triples[x].split(', ')])
        tgt_triples.append(row)
        
        if index < 5:
            print("bert2post:{}".format(bert2post))
            print("src_tokens:{}".format(src_tokens))
            print("src_word_insts:{}".format(src_word_insts[-1]))
            print("src_triples:{}".format(src_triples[-1]))
            print("match_triples:{}".format(match_triples[-1]))
            print("tgt_triples:{}".format(tgt_triples[-1]))
            print("tgt_word_insts:{}".format(tgt_word_insts[-1]))
            print("entitys:{}".format(entitys[-1]))
    return src_word_insts, tgt_word_insts, src_triples, tgt_triples, entitys, match_triples, bert2post_index, post_triple_index

def load_vocab_from_bert():
    def load_vocab(vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = {}
        index = 0
        with open(vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab
    word2idx = {
        CLS_WORD: CLS,
        SEP_WORD: SEP,
        PAD_WORD: PAD,
        UNK_WORD: UNK}
    vocab = load_vocab('uncased_L-12_H-768_A-12/vocab.txt')
    #ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
    return vocab

def build_entity_vocab(data_dir):
    print("Creating entity vocabulary...")
    entity_list = ['_NONE', '_PAD_H', '_PAD_R', '_PAD_T', '_NAF_H', '_NAF_R', '_NAF_T']
    with open('%s/entity.txt' % data_dir) as f:
        for i, line in enumerate(f):
            e = line.strip()
            entity_list.append(e)
    print("Creating relation vocabulary...")
    relation_list = []
    with open('%s/relation.txt' % data_dir) as f:
        for i, line in enumerate(f):
            r = line.strip()
            relation_list.append(r)
    entity_relation = entity_list + relation_list
    vocab = {}
    for index, w in enumerate(entity_relation):
        vocab[w] = index
    print('entity vocab len:{}'.format(len(vocab)))
    return vocab
    
def build_entity_emb(data_dir, trans='transE'):
    print("Loading entity vectors...")
    entity_embed = []
    emb_size = None
    with open('%s/entity_%s.txt' % (data_dir, trans)) as f:
        for i, line in enumerate(f):
            s = line.strip().split('\t')
            if emb_size is None:
                emb_size = len(s)
            entity_embed.append([i for i in map(float, s)])

    print("Loading relation vectors...")
    relation_embed = []
    with open('%s/relation_%s.txt' % (data_dir, trans)) as f:
        for i, line in enumerate(f):
            s = line.strip().split('\t')
            relation_embed.append([i for i in map(float, s)])
    print("first 7 pad random sample")
    
    first7_embed = [[0.0] * emb_size]*7
    entity_relation_embed = np.array(first7_embed + entity_embed+relation_embed, dtype=np.float32)
    torch.save(entity_relation_embed, os.path.join(data_dir, 'entity_emb.torch'))
    print("save entity_emb in entity_emb.pt")
    return entity_relation_embed

def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', default='data/')
    parser.add_argument('-save_data', default='data/triple_test.pt')
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=100)
    parser.add_argument('-max_tgt_len', type=int, default=60)
    parser.add_argument('-min_word_count', type=int, default=10)
    parser.add_argument('-keep_case', action='store_false')
    parser.add_argument('-do_lower_case', type=bool, default=True)
    parser.add_argument('-bert_vocab_file', type=str, default="uncased_L-12_H-768_A-12/vocab.txt")
    parser.add_argument('-share_vocab', type=bool, default=True)
    parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len + 2 # include the <s> and </s>
    opt.max_token_tgt_len = opt.max_tgt_len + 2
    
    entity_vocab = build_entity_vocab(opt.data_dir)
    
    # Training set
    train_src_word_insts, train_tgt_word_insts, train_src_triples, train_tgt_triples, train_entitys, train_match_triples, train_bert2post, train_post_triple = read_instances_from_data_file(
        opt, entity_vocab, istrain=True)
    print('train_seq')
    print(train_src_word_insts[0])
    checkLength(train_src_word_insts)
    print(train_tgt_word_insts[0])
    checkLength(train_tgt_word_insts)
    assert len(train_src_word_insts) == len(train_tgt_word_insts)

    # Validation set
    valid_src_word_insts, valid_tgt_word_insts, valid_src_triples, valid_tgt_triples, valid_entitys, valid_match_triples, valid_bert2post, valid_post_triple = read_instances_from_data_file(opt, entity_vocab, istrain=False)
    print('valid_seq')
    print(valid_src_word_insts[-1])
    checkLength(valid_src_word_insts)
    print(valid_tgt_word_insts[-1])
    checkLength(valid_tgt_word_insts)
    
    assert len(valid_src_word_insts) == len(valid_tgt_word_insts)

    # Build vocabulary
    print('[Info] Build shared vocabulary for source and target.')
    ############################################################
    word2idx = load_vocab_from_bert()
    src_word2idx = tgt_word2idx = word2idx

    ####build entity_emb
    entity_emb = build_entity_emb(opt.data_dir)

    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx,
            'entity':entity_vocab},
        'train': {
            'src': train_src_word_insts,
            'tgt': train_tgt_word_insts,
            'entitys': train_entitys,
            'src_triples': train_src_triples,
            'match_triples': train_match_triples,
            'tgt_triples': train_tgt_triples,
            'bert2post': train_bert2post,
            'post2triple': train_post_triple
            },
        'valid': {
            'src': valid_src_word_insts,
            'tgt': valid_tgt_word_insts,
            'entitys': valid_entitys,
            'src_triples': valid_src_triples,
            'match_triples': valid_match_triples,
            'tgt_triples': valid_tgt_triples,
            'bert2post': valid_bert2post,
            'post2triple': valid_post_triple
            }}

    print("max len of train entitys is {}".format(max([len(i) for i in train_entitys])))
    print("max len of valid entitys is {}".format(max([len(i) for i in valid_entitys])))
    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()
