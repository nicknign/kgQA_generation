#!/bin/sh

SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)

cd $SHELL_FOLDER

rm /shard2

/workspace/python3_conda -m torch.distributed.launch --nproc_per_node 4 train_triple.py \
  -url /shard2 \
  -world_size 4 \
  -batch_size 24 \
  -epoch 10 \
  -lr 5e-5 \
  -log_step 1000 \
  -n_layers 2 \
  -data data/triple_30w.pt \
  -save_model models_triple_2bert/kg_triple_30w \
  -save_mode all \
  --bert_model uncased_L-12_H-768_A-12

/workspace/python3_conda test_triple_mcard.py models_triple_2bert
