#!/bin/sh

CUDA_VISIBLE_DEVICES=0 \
python3 train.py \
    --output_dir=output/bfm_shadegan_noview \
    --curriculum=BFM_ShadeGAN_noview \
    --load_dir=weights/pretrain/bfm_noview/pretrain5k- \
    --set_step=5000
