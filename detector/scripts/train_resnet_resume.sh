#!/usr/bin/env bash
GPU_ID=0
BATCH_SIZE=1
WORKER_NUMBER=1
LEARNING_RATE=0.001
DECAY_STEP=5

CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
    --dataset pascal_voc --net res101 \
    --bs $BATCH_SIZE --nw $WORKER_NUMBER \
    --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
    --r True \
    --checksession 1 \
    --checkepoch 16 \
    --checkpoint 18197 \
    --cuda