# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

import os
import numpy as np
import argparse
from nltk.translate.bleu_score import corpus_bleu

def get_data(filepath):
    f=open(filepath,encoding='utf-8')
    sentences=[]
    for line in f.readlines():
        sentences.append(line.strip())
    return sentences
def get_src_data(filepath):
    f=open(filepath,encoding='utf-8')
    sentences=[]
    for line in f.readlines():
        sent = line.split('  +++  ')
        sentences.append(sent[1])
    return sentences

def get_tgt_data(filepath):
    f=open(filepath,encoding='utf-8')
    sentences=[]
    for line in f.readlines():
        sent = line.split('  +++  ')
        sentences.append(sent[2])
    return sentences

def get_ans_data(filepath):
    f=open(filepath,encoding='utf-8')
    sentences=[]
    for line in f.readlines():
        sent = line.split('  +++  ')
        sentences.append(sent[3])
    return sentences

def eval(data,pred,out):
    list_of_refs, hypotheses = [], []
    get_data(data)
    sources= get_src_data(data)
    targets = get_tgt_data(data)
    preds = get_data(pred)
    ans = get_ans_data(data)
    ### Write to file
    fout=open(out,'w',encoding='utf-8')
    for i in range(0,len(sources)):  # sentence-wise
        #preds[i]=preds[i].strip('</s>')
        fout.write("- source: " + sources[i] + "\n")
        fout.write("- expected: " + targets[i] + "\n")
        fout.write("- ans: " + ans[i] + "\n")
        fout.write("- got: " + preds[i] + "\n\n")
        fout.flush()

        # bleu score
        ref = targets[i].lower().split()
        hypothesis = preds[i].lower().split()
        if len(ref) > 3 and len(hypothesis) > 3:
            list_of_refs.append([ref])
            hypotheses.append(hypothesis)

            ## Calculate bleu score
    score = corpus_bleu(list_of_refs, hypotheses)
    fout.write("Bleu Score = " + str(100 * score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', required=True)
    parser.add_argument('-pred', required=True)
    parser.add_argument('-out', required=True)
    opt = parser.parse_args()
    data = opt.data
    pred = opt.pred
    out = opt.out
    eval(data,pred,out)
    print("Done")
