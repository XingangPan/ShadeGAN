import os
import shutil
import torch
import math

from torch_fidelity import calculate_metrics
from torchvision.utils import save_image
from tqdm import tqdm
import copy
import argparse
import curriculums
import shutil
from mmcv.runner import init_dist

from generators import generators
from siren import siren
from generators.utils import LSampler
import curriculums


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('generator_file', type=str)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--real_image_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='temp')
    parser.add_argument('--num_images', type=int, default=2048)
    parser.add_argument('--max_batch_size', type=int, default=94800000)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--curriculum', type=str, default='CelebA_ShadeGAN_view')
    parser.add_argument('--num_steps', type=int, default=12)
    parser.add_argument('--delta', type=float, default=-1)
    parser.add_argument('--local_rank', type=int, default=0)

    opt = parser.parse_args()

    if opt.distributed:
        import torch.distributed as dist
        init_dist('pytorch', backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size = 0, 1

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(rank)
    device = torch.device(rank)

    if rank == 0:
        if os.path.exists(opt.output_dir) and os.path.isdir(opt.output_dir):
            shutil.rmtree(opt.output_dir)
        os.makedirs(opt.output_dir, exist_ok=False)

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)
    
    ldist = LSampler(device=device, dataset=metadata['dataset'])
    SIREN = getattr(siren, curriculum['model'])
    generator = getattr(generators, curriculum['generator'])(SIREN, curriculum['latent_dim'], curriculum['shading'],
                                                             curriculum['view_condition'], curriculum['light_condition'],
                                                             curriculum['surf_track'], ldist=ldist).to(device)

    if not opt.ema:
        generator.load_state_dict(torch.load(opt.generator_file, map_location=device), strict=False)

    generator.set_device(device)
    if opt.ema:
        ema_file = opt.generator_file.split('generator')[0] + 'ema.pth'
        ema = torch.load(ema_file, map_location=device)
        ema.copy_to(generator.parameters())
    generator.eval()

    options_dict = copy.deepcopy(metadata)
    options_dict['img_size'] = 128
    options_dict['num_steps'] = opt.num_steps
    options_dict['psi'] = 1
    options_dict['nerf_noise'] = 0
    options_dict['delta'] = opt.delta

    opt.num_images = opt.num_images // world_size
    start = rank * opt.num_images

    for img_counter in tqdm(range(start,start+opt.num_images)):
        z = torch.randn(1, 256, device=device)
        l = ldist.sample(1)

        with torch.no_grad():
            img = generator.staged_forward(z, l, max_batch_size=opt.max_batch_size, **options_dict)['rgb'].to(device)
            save_image(img, os.path.join(opt.output_dir, f'{img_counter:0>5}.jpg'), normalize=True, range=(-1, 1))

    if rank == 0:
        metrics_dict = calculate_metrics(opt.output_dir, opt.real_image_dir, cuda=True, isc=True, fid=True, kid=True, verbose=False)
        print(metrics_dict)