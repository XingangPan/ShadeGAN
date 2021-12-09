
CUDA_VISIBLE_DEVICES=0 python render.py \
    weights/pretrain/celeba_noview/generator.pth \
    --output_dir imgs \
    --curriculum CelebA_ShadeGAN_noview \
    --seeds 0 5 8 43 \
    --num_steps 12 \
    --sample_dist fixed \
    --psi 0.5 \
    --delta 0.06491 \
    --image_size 256 \
    --ema \
    --rotate \
    --relight
