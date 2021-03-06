#!/usr/bin/env python

import sys
import os
import json

from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from collections import defaultdict
from argparse import ArgumentParser

class Eval:
    def __init__(self, gts, res):
        self.gts = gts
        self.res = res

    def evaluate(self):
        output = []
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            #(Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            # print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(self.gts, self.res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f"%(m, sc * 100))
                    output.append(sc)
            else:
                print("%s: %0.3f"%(method, score * 100))
                output.append(score)
        return output

def eval(out_file, tgt_file):
    """
        Given a filename, calculate the metric scores for that prediction file

        isDin: boolean value to check whether input file is DirectIn.txt
    """

    with open(out_file, 'r') as infile:
        out = [line[:-1] for line in infile]

    with open(tgt_file, "r") as infile:
        tgt = [line[:-1] for line in infile]


    ## eval
    from eval import Eval
    import json
    from json import encoder
    encoder.FLOAT_REPR = lambda o: format(o, '.4f')

    each_score = []
    
    for idx, (out_, tgt_) in enumerate(zip(out, tgt)):
        res = defaultdict(lambda: [])
        gts = defaultdict(lambda: [])
        res[0] = [out_.encode('utf-8')]
        ## gts 
        gts[0] = [tgt_.encode('utf-8')]
        eval = Eval(gts, res)
        each_score.append(eval.evaluate())
    return each_score

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-out", "--out_file", dest="out_file", default="./output/pred.txt", help="output file to compare")
    parser.add_argument("-tgt", "--tgt_file", dest="tgt_file", default="../data/processed/tgt-test.txt", help="target file")
    args = parser.parse_args()
    print(args.out_file)
    print("scores: \n")
    eval_list = eval(args.out_file, args.tgt_file)
    json.dump(eval_list, open("score.json", "w"), indent=2)


