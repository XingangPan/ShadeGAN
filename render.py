import argparse
import torch
import math
import os
import copy
from tqdm import tqdm
from torchvision.utils import save_image

from generators import generators
from siren import siren
from generators.utils import LSampler
from generators.volumetric_rendering import *
import curriculums

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-7


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--seeds', nargs='+', default=[0, 1, 2])
    parser.add_argument('--output_dir', type=str, default='imgs')
    parser.add_argument('--curriculum', type=str, default='CelebA_ShadeGAN_noview')
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--sample_dist', type=str, default='fixed')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=24)
    parser.add_argument('--psi', type=float, default=0.5)
    parser.add_argument('--delta', type=float, default=0.0325)
    parser.add_argument('--rotate', action='store_true')
    parser.add_argument('--relight', action='store_true')
    parser.add_argument('--ema', action='store_true')
    opt = parser.parse_args()

    os.makedirs(opt.output_dir, exist_ok=True)

    opt.seeds = [int(seed) for seed in opt.seeds]

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)
    
    ldist = LSampler(device=device, dataset=metadata['dataset'])
    SIREN = getattr(siren, metadata['model'])
    generator = getattr(generators, metadata['generator'])(SIREN, metadata['latent_dim'], metadata['shading'],
                                                           metadata['view_condition'], metadata['light_condition'],
                                                           metadata['surf_track'], ldist=ldist).to(device)

    if not opt.ema:
        generator.load_state_dict(torch.load(opt.path, map_location=torch.device(device)), strict=False)
    generator.set_device(device)
    if opt.ema:
        ema_file = opt.path.split('generator')[0] + 'ema.pth'
        ema = torch.load(ema_file)
        ema.copy_to(generator.parameters())
    generator.eval()

    options_dict = copy.deepcopy(metadata)
    options_dict['img_size'] = opt.image_size
    options_dict['num_steps'] = opt.num_steps
    options_dict['psi'] = opt.psi
    options_dict['delta'] = opt.delta
    options_dict['sample_dist'] = opt.sample_dist
    options_dict['lock_view_dependence'] = opt.lock_view_dependence
    options_dict['nerf_noise'] = 0
    
    if opt.rotate:
        face_angles = [-0.5, -0.25, 0., 0.25, 0.5]
    else:
        face_angles = [0.]

    face_angles = [a + options_dict['h_mean'] for a in face_angles]

    if opt.relight:
        dxs = [-1.5, -1, 0, 1, 1.5]  # This controls light angle
        # dxs = [-1.5, -1.0, -0.7, -0.35, 0, 0.35, 0.7, 1.0, 1.5]
    else:
        dxs = [0]
    dy = 0.3

    for seed in tqdm(opt.seeds):
        torch.manual_seed(seed)
        z = torch.randn((1, 256), device=device)
        l = ldist.sample(1)
        for i, yaw in enumerate(face_angles):
            options_dict['h_mean'] = yaw
            options_dict['v_mean'] = math.pi/2
            for j, dx in enumerate(dxs):
                if opt.relight:
                    l = torch.zeros((1,4), device=device)
                    l[:,0].fill_(0.17)
                    l[:,1].fill_(0.42)
                    l[:,2].fill_(dx)
                    l[:,3].fill_(dy)
                with torch.no_grad():
                    results = generator.staged_forward(z, l, rt_normal=True, **options_dict)
                    img, depth_map, normal_map, pose = results['rgb'], results['depth'], results['normal'], results['pose']
                prefix = f"img_seed{seed:05d}_yaw{i}_light{j}"
                im_name = f"{prefix}_{options_dict['num_steps']}.png"
                save_image(img/2+0.5, os.path.join(opt.output_dir, im_name), normalize=False)
                save_image((depth_map-0.88)/0.24, os.path.join(opt.output_dir, f"{prefix}_depth.png"), normalize=True)
                save_image(normal_map/2+0.5, os.path.join(opt.output_dir, f"img_seed{seed:05d}_yaw{i}_normal.png"), normalize=False)
                if metadata['surf_track']:
                    save_image((results['depth_pred']-0.88)/0.24, os.path.join(opt.output_dir, f"{prefix}_depth_pred.png"), normalize=True)
                if metadata['shading']:
                    save_image(results['albedo']/2+0.5, os.path.join(opt.output_dir, f"{prefix}_albedo.png"), normalize=False)
                    shading = results['shading']
                    ambience = l[:,None,:1]/2+0.5
                    diffuse = l[:,None,1:2]/2+0.5
                    diffuse_shading = (shading - ambience) / diffuse
                    save_image(diffuse_shading, os.path.join(opt.output_dir, f"{prefix}_diffuse.png"), normalize=False)
                    save_image(shading / shading.max().item(), os.path.join(opt.output_dir, f"{prefix}_shading.png"), normalize=False)
