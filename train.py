"""Train ShadeGAN. Supports distributed training."""

import argparse
import os
import math
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from generators import generators
from discriminators import discriminators
from siren import siren
from lpips import PerceptualLoss
from generators.utils import LSampler, z_sampler, act_scheduler, rendering_scheduler
import fid_evaluation

import datasets
import curriculums
from tqdm import tqdm
from datetime import datetime
import copy

from torch_ema import ExponentialMovingAverage


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, opt):
    torch.manual_seed(0)

    setup(rank, world_size, opt.port)
    torch.cuda.set_device(rank)
    device = torch.device(rank)

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)

    fixed_z = z_sampler((25, 256), device='cpu', dist=metadata['z_dist'])
    ldist = LSampler(device=device,
                     dataset=metadata['dataset'],
                     mvn_path=metadata['mvn_path'] if 'mvn_path' in metadata else None)
    fixed_l = ldist.sample(25)

    SIREN = getattr(siren, metadata['model'])

    scaler = torch.cuda.amp.GradScaler()

    generator = getattr(generators, metadata['generator'])(SIREN, metadata['latent_dim'], metadata['shading'], \
        metadata['view_condition'], metadata['light_condition'], metadata['surf_track'], ldist=ldist).to(device)
    discriminator = getattr(discriminators, metadata['discriminator'])().to(device)
    ema = ExponentialMovingAverage(generator.parameters(), decay=0.999)
    ema2 = ExponentialMovingAverage(generator.parameters(), decay=0.9999)
    if opt.load_dir != '':
        generator.load_state_dict(torch.load(opt.load_dir + 'generator.pth', map_location=device), strict=False)
        discriminator.load_state_dict(torch.load(opt.load_dir + 'discriminator.pth', map_location=device), strict=False)
        if os.path.isfile(opt.load_dir + 'ema.pth'):
            ema = torch.load(opt.load_dir + 'ema.pth', map_location=device)
            ema2 = torch.load(opt.load_dir + 'ema2.pth', map_location=device)

    generator_ddp = DDP(generator, device_ids=[rank], find_unused_parameters=True)
    discriminator_ddp = DDP(discriminator, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
    generator = generator_ddp.module
    discriminator = discriminator_ddp.module
    if metadata['surf_track']:
        perceptual_loss = PerceptualLoss(model='net-lin', net='vgg', use_gpu=True, gpu_ids=[device])

    if metadata.get('unique_lr', False):
        mapping_network_param_names = [name for name, _ in generator_ddp.module.siren.mapping_network.named_parameters()]
        mapping_network_parameters = [p for n, p in generator_ddp.named_parameters() if n in mapping_network_param_names]
        if metadata['surf_track']:
            surfacenet_param_names = [name for name, _ in generator_ddp.module.surfacenet.named_parameters()]
            surfacenet_parameters = [p for n, p in generator_ddp.named_parameters() if n in surfacenet_param_names]
            generator_parameters = [p for n, p in generator_ddp.named_parameters() if n not in [mapping_network_param_names+surfacenet_param_names]]
            optimizer_G = torch.optim.Adam([{'params': generator_parameters, 'name': 'generator'},
                                            {'params': surfacenet_parameters, 'name': 'surfacenet', 'lr':metadata['gen_lr']*10},
                                            {'params': mapping_network_parameters, 'name': 'mapping_network', 'lr':metadata['gen_lr']*5e-2}],
                                           lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])
        else:  
            generator_parameters = [p for n, p in generator_ddp.named_parameters() if n not in mapping_network_param_names]
            optimizer_G = torch.optim.Adam([{'params': generator_parameters, 'name': 'generator'},
                                            {'params': mapping_network_parameters, 'name': 'mapping_network', 'lr':metadata['gen_lr']*5e-2}],
                                           lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])
    else:
        optimizer_G = torch.optim.Adam(generator_ddp.parameters(), lr=metadata['gen_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    optimizer_D = torch.optim.Adam(discriminator_ddp.parameters(), lr=metadata['disc_lr'], betas=metadata['betas'], weight_decay=metadata['weight_decay'])

    if opt.load_dir != '':
        if os.path.isfile(opt.load_dir + 'optimizer_G.pth'):
            optimizer_G.load_state_dict(torch.load(opt.load_dir + 'optimizer_G.pth', map_location=device))
            optimizer_D.load_state_dict(torch.load(opt.load_dir + 'optimizer_D.pth', map_location=device))
        if not metadata.get('disable_scaler', False):
            scaler.load_state_dict(torch.load(opt.load_dir + 'scaler.pth', map_location=device))

    generator_losses = []
    discriminator_losses = []

    if opt.set_step != None:
        generator.step = opt.set_step
        discriminator.step = opt.set_step

    if metadata.get('disable_scaler', False):
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    generator.set_device(device)

    # ----------
    #  Training
    # ----------

    with open(os.path.join(opt.output_dir, 'options.txt'), 'w') as f:
        f.write(str(opt))
        f.write('\n\n')
        f.write(str(generator))
        f.write('\n\n')
        f.write(str(discriminator))
        f.write('\n\n')
        f.write(str(curriculum))

    torch.manual_seed(rank)
    dataloader = None
    total_progress_bar = tqdm(total = opt.n_epochs, desc = "Total progress", dynamic_ncols=True)
    total_progress_bar.update(discriminator.epoch)
    interior_step_bar = tqdm(dynamic_ncols=True)
    t0 = time.time()
    for _ in range (opt.n_epochs):
        total_progress_bar.update(1)

        metadata = curriculums.extract_metadata(curriculum, discriminator.step)

        if not dataloader or dataloader.batch_size != metadata['batch_size']:
            dataloader, CHANNELS = datasets.get_dataset_distributed(metadata['dataset'],
                                        world_size,
                                        rank,
                                        **metadata)

            step_next_upsample = curriculums.next_upsample_step(curriculum, discriminator.step)
            step_last_upsample = curriculums.last_upsample_step(curriculum, discriminator.step)


            interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
            interior_step_bar.set_description(f"Progress to next stage")
            interior_step_bar.update((discriminator.step - step_last_upsample))

        for i, (imgs, _) in enumerate(dataloader):
            if discriminator.step % opt.model_save_interval == 0 and rank == 0:
                now = datetime.now()
                now = now.strftime("%d--%H:%M--")
                torch.save(ema, os.path.join(opt.output_dir, now + 'ema.pth'))
                torch.save(ema2, os.path.join(opt.output_dir, now + 'ema2.pth'))
                torch.save(generator_ddp.module.state_dict(), os.path.join(opt.output_dir, now + 'generator.pth'))
                torch.save(discriminator_ddp.module.state_dict(), os.path.join(opt.output_dir, now + 'discriminator.pth'))
                torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, now + 'optimizer_G.pth'))
                torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, now + 'optimizer_D.pth'))
                torch.save(scaler.state_dict(), os.path.join(opt.output_dir, now + 'scaler.pth'))
            metadata = curriculums.extract_metadata(curriculum, discriminator.step)

            if dataloader.batch_size != metadata['batch_size']: break

            if metadata['surf_track']:
                metadata['delta'], metadata['num_steps'] = rendering_scheduler(discriminator.step, **metadata)
                lr_ratio = metadata['num_steps'] / metadata['num_steps_max']
            else:
                metadata['delta'] = -1
                lr_ratio = 1

            # Set learning rates
            for param_group in optimizer_G.param_groups:
                if param_group.get('name', None) == 'mapping_network':
                    param_group['lr'] = metadata['gen_lr'] * 5e-2 * lr_ratio
                elif param_group.get('name', None) == 'surfacenet':
                    param_group['lr'] = metadata['gen_lr'] * 10
                else:
                    param_group['lr'] = metadata['gen_lr'] * lr_ratio
                param_group['betas'] = metadata['betas']
                param_group['weight_decay'] = metadata['weight_decay']
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = metadata['disc_lr']
                param_group['betas'] = metadata['betas']
                param_group['weight_decay'] = metadata['weight_decay']

            if scaler.get_scale() < 1:
                scaler.update(1.)

            generator_ddp.train()
            discriminator_ddp.train()

            alpha = min(1, (discriminator.step - step_last_upsample) / (metadata['fade_steps']))

            real_imgs = imgs.to(device, non_blocking=True)

            metadata['nerf_noise'] = max(0, 1. - discriminator.step/5000.)
            metadata['l_ratio'] = act_scheduler(discriminator.step)

            # TRAIN DISCRIMINATOR
            with torch.cuda.amp.autocast():
                # Generate images for discriminator training
                with torch.no_grad():
                    z = z_sampler((real_imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])
                    l = ldist.sample(real_imgs.shape[0])
                    split_batch_size = z.shape[0] // metadata['batch_split']
                    gen_imgs = []
                    gen_positions = []
                    for split in range(metadata['batch_split']):
                        subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                        subset_l = l[split * split_batch_size:(split+1) * split_batch_size]
                        results = generator_ddp(subset_z, subset_l, **metadata)
                        g_imgs, g_pos = results['rgb'], results['pose']

                        gen_imgs.append(g_imgs)
                        gen_positions.append(g_pos)

                    gen_imgs = torch.cat(gen_imgs, axis=0)
                    gen_positions = torch.cat(gen_positions, axis=0)

                real_imgs.requires_grad = True
                r_preds, _, _ = discriminator_ddp(real_imgs, alpha, **metadata)

            if metadata['r1_lambda'] > 0:
                # Gradient penalty
                grad_real = torch.autograd.grad(outputs=scaler.scale(r_preds.sum()), inputs=real_imgs, create_graph=True)
                inv_scale = 1./scaler.get_scale()
                grad_real = [p * inv_scale for p in grad_real][0]
            with torch.cuda.amp.autocast():
                if metadata['r1_lambda'] > 0:
                    grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    grad_penalty = 0.5 * metadata['r1_lambda'] * grad_penalty
                else:
                    grad_penalty = 0

                g_preds, g_pred_latent, g_pred_position = discriminator_ddp(gen_imgs, alpha, **metadata)
                if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:
                    latent_penalty = torch.nn.MSELoss()(g_pred_latent, z) * metadata['z_lambda']
                    position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * metadata['pos_lambda']
                    identity_penalty = latent_penalty + position_penalty
                else:
                    identity_penalty=0

                d_loss = torch.nn.functional.softplus(g_preds).mean() + torch.nn.functional.softplus(-r_preds).mean() + grad_penalty + identity_penalty
                discriminator_losses.append(d_loss.item())

            optimizer_D.zero_grad()
            scaler.scale(d_loss).backward()
            scaler.unscale_(optimizer_D)
            torch.nn.utils.clip_grad_norm_(discriminator_ddp.parameters(), metadata['grad_clip'])
            scaler.step(optimizer_D)


            # TRAIN GENERATOR
            z = z_sampler((imgs.shape[0], metadata['latent_dim']), device=device, dist=metadata['z_dist'])
            l = ldist.sample(imgs.shape[0])

            split_batch_size = z.shape[0] // metadata['batch_split']

            for split in range(metadata['batch_split']):
                with torch.cuda.amp.autocast():
                    subset_z = z[split * split_batch_size:(split+1) * split_batch_size]
                    subset_l = l[split * split_batch_size:(split+1) * split_batch_size]
                    results = generator_ddp(subset_z, subset_l, **metadata)
                    gen_imgs, gen_positions = results['rgb'], results['pose']
                    depth_pred, depth = results['depth_pred'], results['depth'].detach()
                    g_preds, g_pred_latent, g_pred_position = discriminator_ddp(gen_imgs, alpha, **metadata)

                    topk_percentage = max(0.99 ** (discriminator.step/metadata['topk_interval']), metadata['topk_v']) if 'topk_interval' in metadata and 'topk_v' in metadata else 1
                    topk_num = math.ceil(topk_percentage * g_preds.shape[0])

                    g_preds = torch.topk(g_preds, topk_num, dim=0).values

                    if metadata['z_lambda'] > 0 or metadata['pos_lambda'] > 0:
                        latent_penalty = torch.nn.MSELoss()(g_pred_latent, subset_z) * metadata['z_lambda']
                        position_penalty = torch.nn.MSELoss()(g_pred_position, gen_positions) * metadata['pos_lambda']
                        identity_penalty = latent_penalty + position_penalty
                    else:
                        identity_penalty = 0

                    if metadata['surf_track']:
                        depth_pred_norm = ((depth_pred - metadata['ray_start']) / (metadata['ray_end'] - metadata['ray_start'])) * 2 - 1
                        depth_norm = ((depth - metadata['ray_start']) / (metadata['ray_end'] - metadata['ray_start'])) * 2 - 1
                        depth_loss = F.l1_loss(depth_pred_norm, depth_norm) + torch.mean(perceptual_loss(depth_pred_norm.unsqueeze(1).repeat(1,3,1,1), depth_norm.unsqueeze(1).repeat(1,3,1,1)))
                    else:
                        depth_loss = torch.zeros(1).to(device)

                    g_loss = torch.nn.functional.softplus(-g_preds).mean() + identity_penalty
                    generator_losses.append(g_loss.item())

                scaler.scale(g_loss + depth_loss).backward()

            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator_ddp.parameters(), metadata.get('grad_clip', 0.3))
            scaler.step(optimizer_G)
            scaler.update()
            optimizer_G.zero_grad()
            ema.update(generator_ddp.parameters())
            ema2.update(generator_ddp.parameters())


            if rank == 0:
                interior_step_bar.update(1)
                if i%10 == 0:
                    tqdm.write(f"Epoch: {discriminator.epoch}  Step: {discriminator.step}  D: {d_loss.item():.3f}  G: {g_loss.item():.3f}  Depth: {depth_loss.item():.3f}  STD_r: {results['depth_std'].mean().item():.4f}  Delta: {metadata['delta']:.4f}  Num step: {metadata['num_steps']}  Alpha: {alpha:.2f}  Img: {metadata['img_size']}  Batch: {metadata['batch_size']}  TopK: {topk_num}  Scale: {scaler.get_scale()}")

                if discriminator.step % opt.sample_interval == 0:
                    generator_ddp.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['img_size'] = 128
                            results = generator_ddp.module.staged_forward(fixed_z.to(device), fixed_l, **copied_metadata)
                    save_image(results['rgb'][:25]/2+0.5, os.path.join(opt.output_dir, f"{discriminator.step}_fixed.jpg"), nrow=5, normalize=False)
                    save_image(results['depth'][:25].unsqueeze(1), os.path.join(opt.output_dir, f"{discriminator.step}_fixed_depth.jpg"), nrow=5, normalize=True)
                    if metadata['surf_track']:
                        save_image(results['depth_pred'][:25].unsqueeze(1), os.path.join(opt.output_dir, f"{discriminator.step}_fixed_depth_pred.jpg"), nrow=5, normalize=True)
                    if metadata['shading']:
                        save_image(results['normal'][:25]/2+0.5, os.path.join(opt.output_dir, f"{discriminator.step}_fixed_normal.jpg"), nrow=5, normalize=False)
                        save_image(results['shading'][:25].unsqueeze(1), os.path.join(opt.output_dir, f"{discriminator.step}_fixed_shading.jpg"), nrow=5, normalize=False)
                        save_image(results['shading'][:25].unsqueeze(1), os.path.join(opt.output_dir, f"{discriminator.step}_fixed_shading2.jpg"), nrow=5, normalize=True)
                        save_image(results['albedo'][:25]/2+0.5, os.path.join(opt.output_dir, f"{discriminator.step}_fixed_albedo.jpg"), nrow=5, normalize=False)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z.to(device), fixed_l, **copied_metadata)['rgb']
                    save_image(gen_imgs[:25]/2+0.5, os.path.join(opt.output_dir, f"{discriminator.step}_tilted.jpg"), nrow=5, normalize=False)

                    ema.store(generator_ddp.parameters())
                    ema.copy_to(generator_ddp.parameters())
                    generator_ddp.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z.to(device), fixed_l, **copied_metadata)['rgb']
                    save_image(gen_imgs[:25]/2+0.5, os.path.join(opt.output_dir, f"{discriminator.step}_fixed_ema.jpg"), nrow=5, normalize=False)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['h_mean'] += 0.5
                            copied_metadata['img_size'] = 128
                            gen_imgs = generator_ddp.module.staged_forward(fixed_z.to(device), fixed_l, **copied_metadata)['rgb']
                    save_image(gen_imgs[:25]/2+0.5, os.path.join(opt.output_dir, f"{discriminator.step}_tilted_ema.jpg"), nrow=5, normalize=False)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            copied_metadata = copy.deepcopy(metadata)
                            copied_metadata['img_size'] = 128
                            copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                            copied_metadata['psi'] = 0.7
                            gen_imgs = generator_ddp.module.staged_forward(torch.randn_like(fixed_z).to(device), fixed_l, **copied_metadata)['rgb']
                    save_image(gen_imgs[:25]/2+0.5, os.path.join(opt.output_dir, f"{discriminator.step}_random.jpg"), nrow=5, normalize=False)

                    ema.restore(generator_ddp.parameters())

                if discriminator.step % opt.sample_interval == 0:
                    torch.save(ema, os.path.join(opt.output_dir, 'ema.pth'))
                    torch.save(ema2, os.path.join(opt.output_dir, 'ema2.pth'))
                    torch.save(generator_ddp.module.state_dict(), os.path.join(opt.output_dir, 'generator.pth'))
                    torch.save(discriminator_ddp.module.state_dict(), os.path.join(opt.output_dir, 'discriminator.pth'))
                    torch.save(optimizer_G.state_dict(), os.path.join(opt.output_dir, 'optimizer_G.pth'))
                    torch.save(optimizer_D.state_dict(), os.path.join(opt.output_dir, 'optimizer_D.pth'))
                    torch.save(scaler.state_dict(), os.path.join(opt.output_dir, 'scaler.pth'))
                    torch.save(generator_losses, os.path.join(opt.output_dir, 'generator.losses'))
                    torch.save(discriminator_losses, os.path.join(opt.output_dir, 'discriminator.losses'))

            if opt.eval_freq > 0 and discriminator.step % opt.eval_freq == 0:
                generated_dir = os.path.join(opt.output_dir, 'evaluation/generated')

                if rank == 0:
                    fid_evaluation.setup_evaluation(metadata['dataset'], metadata['dataset_path'], generated_dir, target_size=128)
                dist.barrier()
                ema.store(generator_ddp.parameters())
                ema.copy_to(generator_ddp.parameters())
                generator_ddp.eval()
                fid_evaluation.output_images(generator_ddp, metadata, rank, world_size, generated_dir)
                ema.restore(generator_ddp.parameters())
                dist.barrier()
                if rank == 0:
                    fid = fid_evaluation.calculate_fid(metadata['dataset'], generated_dir, target_size=128)
                    with open(os.path.join(opt.output_dir, f'fid.txt'), 'a') as f:
                        f.write(f'\n{discriminator.step}:{fid}')
                    with open(os.path.join(opt.output_dir, f'time.txt'), 'a') as f:
                        f.write(f'\n{discriminator.step}:{(time.time() - t0)/3600}')
                    with open(os.path.join(opt.output_dir, f'render.txt'), 'a') as f:
                        delta, numstep = metadata['delta'], metadata['num_steps']
                        f.write(f'\n{discriminator.step}:{delta} {numstep}')

                torch.cuda.empty_cache()

            discriminator.step += 1
            generator.step += 1
        discriminator.epoch += 1
        generator.epoch += 1

    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
    parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='debug')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--curriculum', type=str, required=True)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--port', type=str, default='12355')
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=5000)

    opt = parser.parse_args()
    print(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)
