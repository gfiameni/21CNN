#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
#export OMP_NUM_THREADS=6
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=ipogif0
export NCCL_IB_CUDA_SUPPORT=1
####python large_run.py \
horovodrun -np 4 -H localhost:4 python large_run.py \
        --N_walker 10000 \
        --N_slice 4 \
        --N_noise 1 \
        --noise_rolling 1 \
        --model playground.ConvRNN3D_simple \
        --model_type RNN \
        --HyperparameterIndex 10 \
        --epochs 10 \
        --max_epochs 1000 \
        --gpus 4 \
	--warmup 10 \
        --data_location /scratch/d.prelogovic/data/SKA_1000/ \
	--saving_location /scratch/d.prelogovic/runs/models/ \
        --logs_location /scratch/d.prelogovic/runs/logs/ \
	--verbose 1 \
	--workers 1 \
	--load_all 1 \
