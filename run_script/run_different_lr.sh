#!/usr/bin/env bash

EXP_ID_BASE='test_FEMNIST_fiaelt_lr'
DATASET='FEMNIST'
NUM_ROUND=200
BATCH_SIZE=10
NUM_EPOCH=20
CLIENT_PER_ROUND=-1
MODEL='ann'

cd ../

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for LEARNING_RATE in 0.012
do
    EXP_ID=$EXP_ID_BASE$LEARNING_RATE
    python3  -u main.py --dataset=$DATASET --optimizer='fiaelt' --exp_id=$EXP_ID  \
                --learning_rate=$LEARNING_RATE --num_rounds=$NUM_ROUND \
                --clients_per_round=$CLIENT_PER_ROUND --eval_every=1 \
                --batch_size=$BATCH_SIZE --num_epochs=$NUM_EPOCH --model=$MODEL
done