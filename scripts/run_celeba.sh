#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 train.py \
    --output_dir=output/celeba_shadegan_noview \
    --curriculum=CelebA_ShadeGAN_noview
