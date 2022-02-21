###The code for paper: 《IMPROVING DIALOGUE RESPONSE GENERATION VIA KNOWLEDGE GRAPH FILTER》

steps:
1. please download The English Reddit datasets according to the original paper: 《Commonsense knowledge aware conversation generation with graph attention》.
2. dowmload the Bert-base model, and put into directory 'uncased_L-12_H-768_A-12'.
2. use 'preprocess_entity_words.py' to generate datasets.
3. run 'train_triple_2bert.sh' to train and test model.


We did our expriements on 8 V100 GPU cards with batch size of 48.
The 'train_triple_2bert.sh' provided here use 4 GPU cards with batch size of 24.
