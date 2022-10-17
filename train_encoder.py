import argparse
import torch
import math
import os
import torch.nn.functional as F

from generators import generators
from siren import siren
from generators.volumetric_rendering import *
from generators.utils import LSampler
from discriminators import discriminators
import curriculums

EPS = 1e-7


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_G', type=str)
    parser.add_argument('path_D', type=str)
    parser.add_argument('--seeds', nargs='+', default=[0, 1, 2])
    parser.add_argument('--output_dir', type=str, default='imgs')
    parser.add_argument('--curriculum', type=str, default='CelebA_ShadeGAN_noview')
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--delta', type=float, default=0.0325)
    parser.add_argument('--ema', action='store_true')
    opt = parser.parse_args()
    
    os.makedirs(opt.output_dir, exist_ok=True)

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)

    device = torch.device('cuda')

    SIREN = getattr(siren, metadata['model'])
    ldist = LSampler(device=device, dataset=metadata['dataset'])
    generator = getattr(generators, metadata['generator'])(SIREN, metadata['latent_dim'], metadata['shading'],
                                                           metadata['view_condition'], metadata['light_condition'],
                                                           metadata['surf_track'], ldist=ldist).to(device)
    encoder = getattr(discriminators, metadata['discriminator'])(inversion=True).to(device)

    generator.load_state_dict(torch.load(opt.path_G, map_location=torch.device(device)), strict=False)
    encoder.load_state_dict(torch.load(opt.path_D, map_location=torch.device(device)), strict=False)

    if opt.ema:
        ema_file = opt.path_G.split('generator')[0] + 'ema.pth'
        ema = torch.load(ema_file, map_location=torch.device(device))
        ema.copy_to(generator.parameters())

    generator.set_device(device)
    generator.eval()
    encoder.train()

    options_dict = {
        'num_steps':6,
        'img_size':64,
        'hierarchical_sample':True,
        'psi':1,
        'ray_start':0.88,
        'ray_end':1.12,
        'v_stddev': 0.155,
        'h_stddev': 0.3,
        'sample_dist': 'gaussian',
        'h_mean': 0 + math.pi/2,
        'v_mean': 0 + math.pi/2,
        'fov': 12,
        'lock_view_dependence': opt.lock_view_dependence,
        'white_back':False,
        'last_back': True,
        'clamp_mode': 'relu',
        'nerf_noise': 0,
        'delta': opt.delta,
    }
    
    batch_size = 4  # use 8 if you have enough GPU memory
    lr=2e-4
    save_name = opt.output_dir + '/encoder.pth'

    optimizer_d = torch.optim.Adam(
        encoder.parameters(),
        lr=lr, betas=metadata['betas'], weight_decay=0)

    for i in range(10000):
        if i in [5000, 8000]:
            for param_group in optimizer_d.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
        if i % 100 == 0:
            torch.save(encoder.state_dict(), save_name)

        z = torch.randn((batch_size, 256), device=device).to(generator.device)
        l = ldist.sample(batch_size).to(generator.device)

        with torch.no_grad():
            raw_frequencies, raw_phase_shifts = generator.siren.mapping_network(z)
            results = generator.forward_with_frequencies(z, raw_frequencies, raw_phase_shifts, l, **options_dict)
            img, pose = results['rgb'], results['pose']
        _, latent, pose_pred, light, freq, phase = encoder(img, 1, **options_dict)
        img_rec = generator.forward_with_frequencies(latent, freq, phase, l, pose=pose_pred, **options_dict)['rgb']
        loss_z = F.mse_loss(latent, z)
        loss_p = F.mse_loss(pose_pred, pose)
        loss_l = F.mse_loss(light, l)
        loss_im = F.mse_loss(img_rec, img)
        loss_freq = F.mse_loss(freq, raw_frequencies)
        loss_phase = F.mse_loss(phase, raw_phase_shifts)
        loss = loss_p + loss_im + loss_freq + loss_phase

        if i % 10 == 0:
            print(f'{i}: loss {loss.item():.4f}')

        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

    torch.save(encoder.state_dict(), save_name)
