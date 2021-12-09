#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 train.py \
    --output_dir=output/cats_shadegan \
    --curriculum=CATS_ShadeGAN \
    --eval_freq=2000 \
    --model_save_interval=2000

