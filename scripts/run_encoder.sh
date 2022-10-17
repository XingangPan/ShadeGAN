python train_encoder.py \
    weights/pretrain/celeba_noview/generator.pth \
    weights/pretrain/celeba_noview/discriminator.pth \
    --curriculum CelebA_ShadeGAN_noview \
    --seeds 0 \
    --output_dir weights/pretrain/celeba_noview \
    --delta 0.06491 \
    --ema
