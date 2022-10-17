python -u inversion.py \
    weights/pretrain/celeba_noview/generator.pth \
    weights/pretrain/celeba_noview/encoder.pth \
    --curriculum CelebA_ShadeGAN_noview \
    --output_dir inversion/shadegan_noview \
    --img_list inversion/list.txt \
    --root inversion \
    --reg_lambda 0.0005 \
    --lr 0.005 \
    --iterations 200 300 350 400 \
    --delta 0.06491 \
    --ema
