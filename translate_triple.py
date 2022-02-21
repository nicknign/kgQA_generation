''' Translate input text with trained model. '''

import torch
import torch.utils.data
import argparse
from tqdm import tqdm
import re,os
import transformer.Constants as Constants
from dataset_entity import collate_fn, TranslationDataset, paired_collate_fn,test_collate_fn_ans
from transformer.Translator_triple_entity_2bert import Translator
import tokenization
import numpy as np
import time
import json

PAD = 0
UNK = 100
CLS = 101
SEP = 102

PAD_WORD = '[PAD]'
UNK_WORD = '[UNK]'
CLS_WORD = '[CLS]'
SEP_WORD = '[SEP]'

SPE_STR = '+++'

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
   

##########################################
# return triples: [[head, relation, tail], [head, relation, tail], ...]
def extract_triple(kb, example, entity_vocab):
    triples = []
    #print("example:{}".format(example))
    if example['movie']:
        for m in example['movie']:
            movie_id = m[1]
            movie_info = kb['movie'][movie_id]
            triple = [[movie_id, 'direct_by', i] for i in movie_info['director']]
            triples.extend(triple)
            triple = [[movie_id, 'act_by', i] for i in movie_info['actor']]
            triples.extend(triple)
            if movie_info['release_time']:
                triple = [[movie_id, 'release_time_is', movie_info['release_time']]]
                triples.extend(triple)

    if example['celebrity']:
        for m in example['celebrity']:
            celebrity_id = m[1]
            celebrity_info = kb['celebrity'][celebrity_id]
            triple = [[i, 'direct_by', celebrity_id] for i in celebrity_info['direct']]
            triples.extend(triple)
            triple = [[i, 'act_by', celebrity_id] for i in celebrity_info['act']]
            triples.extend(triple)

    if example['time']:
        for m in example['time']:
            time_info = kb['time'][m]
            triple = [[i, 'release_time_is', m] for i in time_info['release_movie']]
            triples.extend(triple)
    
    #去重
    #print("triple:{}".format(triples))
    n_triples = [x for x in triples if triples.count(x) == 1]
    #print("n_triples:{}".format(n_triples))
    triple_ids = [[entity_vocab[i] for i in j] for j in n_triples]
    #print(triple_ids[:10])
    return triple_ids


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
    data_file = '{}/testset.txt'.format(opt.data_dir)
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
        entities_line = '|'.join(entities)
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
        
        #onelline_row = sum(row, [])
        #assert len(onelline_row) == len(match_inst)

        #response_triples
        row = []
        for x in example['response_triples']:
            if x == -1:
                continue
            else:
                row.append([entity_vocab.get(i, 0) for i in csk_triples[x].split(', ')])
        tgt_triples.append(row)
        
    return src_word_insts, tgt_word_insts, src_triples, tgt_triples, entitys, match_triples, bert2post_index, post_triple_index


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')
    parser.add_argument('-data_dir', default='data/')
    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-src', required=True,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-vocab', required=True,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-output', required=True,default='./output/',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=12,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true',default=False)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    logger.info(opt.model)

    logger.info("load vocab:{}".format(opt.vocab))
    preprocess_data = torch.load(opt.vocab)
    preprocess_settings = preprocess_data['settings']
    # Prepare DataLoader
    entity_vocab = preprocess_data['dict']['entity']
    if not os.path.exists("data/testprecess.pt"):
        test_src_word_insts, test_tgt_word_insts, test_src_triples, test_tgt_triples, entitys, _, _, _ = read_instances_from_data_file(preprocess_settings, entity_vocab, opt.src)
        save_data = {"settings":preprocess_settings,
                    "test_src_word_insts":test_src_word_insts,
                    "test_src_triples":test_src_triples,
                    "entitys": entitys}
        torch.save(save_data, "data/testprecess.pt")
    else:
        print("load from file: data/testprecess.pt")
        save_data = torch.load("data/testprecess.pt")
        test_src_word_insts = save_data["test_src_word_insts"]
        test_src_triples = save_data["test_src_triples"]
        entitys = save_data["entitys"]

    logger.info(test_src_word_insts[0])
    logger.info(len(test_src_word_insts))

    preprocess_data['settings'].entity_max = 500
    opt.entity_max = 500

    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            setting=preprocess_data['settings'],
            src_word2idx=preprocess_data['dict']['src'],
            tgt_word2idx=preprocess_data['dict']['tgt'],
            entity_word2idx=preprocess_data['dict']['entity'],
            src_insts=test_src_word_insts,
            src_triples=test_src_triples,
            entitys=entitys
            ),
        num_workers=0,
        batch_size=opt.batch_size,
        collate_fn=test_collate_fn_ans)

    entity_emb = torch.load('data/entity_emb.torch')
    #entity_emb = None

    translator = Translator(opt,
                    num_entities=len(entity_vocab),
                    num_trans_units=100,
                    entity_emb=entity_emb)

    with open(opt.output, 'w') as f:
        for batch in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            all_hyp, all_scores, triple_weight = translator.translate_batch(*batch)
            for idx_seqs in all_hyp:
                for idx_seq in idx_seqs:
                    pred_line = ' '.join([test_loader.dataset.tgt_idx2word[idx] for idx in idx_seq])
                    f.write(pred_line + '\n')
    logger.info('[Info] Finished.')

if __name__ == "__main__":
    main()
