import argparse
import torch
import math
import os
import copy
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

from generators import generators
from siren import siren
from generators.volumetric_rendering import *
from generators.utils import LSampler
from generators.math_utils_torch import *
from discriminators import discriminators
import curriculums

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = 1e-7


def load_image(image_path, image_size=128, crop=False):
    if crop:  # for celeba aligned images
        transform = transforms.Compose(
            [
                transforms.Resize(320),
                transforms.CenterCrop(256),
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    image = image * 2 - 1
    return image


def rotate_normal(normal, pose, device):
    n, _, h, w = normal.shape
    pitch, yaw = pose[:,:1], pose[:,1:2]
    camera_origin = pose2origin(device, pitch, yaw, n, 1)
    forward_vector = normalize_vecs(-camera_origin)
    world2cam_matrix = create_world2cam_matrix(forward_vector, camera_origin, device=device)
    transformed_normal = torch.bmm(world2cam_matrix[..., :3, :3], normal.reshape(n, 3, -1)).reshape(n, 3, h, w)
    return transformed_normal


def get_grid(b, H, W, normalize=True):
    if normalize:
        h_range = torch.linspace(-1,1,H)
        w_range = torch.linspace(-1,1,W)
    else:
        h_range = torch.arange(0,H)
        w_range = torch.arange(0,W)
    grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).repeat(b,1,1,1).flip(3).float() # flip h,w to x,y
    return grid


def export_to_obj_string(vertices, normal):
    b, h, w, _ = vertices.shape
    vertices[:,:,:,1:2] = -1*vertices[:,:,:,1:2]  # flip y
    vertices[:,:,:,2:3] = 1-vertices[:,:,:,2:3]  # flip and shift z
    vertices *= 100
    vertices_center = nn.functional.avg_pool2d(vertices.permute(0,3,1,2), 2, stride=1).permute(0,2,3,1)
    vertices = torch.cat([vertices.view(b,h*w,3), vertices_center.view(b,(h-1)*(w-1),3)], 1)

    vertice_textures = get_grid(b, h, w, normalize=True)  # BxHxWx2
    vertice_textures[:,:,:,1:2] = -1*vertice_textures[:,:,:,1:2]  # flip y
    vertice_textures_center = nn.functional.avg_pool2d(vertice_textures.permute(0,3,1,2), 2, stride=1).permute(0,2,3,1)
    vertice_textures = torch.cat([vertice_textures.view(b,h*w,2), vertice_textures_center.view(b,(h-1)*(w-1),2)], 1) /2+0.5  # Bx(H*W)x2, [0,1]

    vertice_normals = normal.clone()
    vertice_normals[:,:,:,0:1] = -1*vertice_normals[:,:,:,0:1]
    vertice_normals_center = nn.functional.avg_pool2d(vertice_normals.permute(0,3,1,2), 2, stride=1).permute(0,2,3,1)
    vertice_normals_center = vertice_normals_center / (vertice_normals_center**2).sum(3, keepdim=True)**0.5
    vertice_normals = torch.cat([vertice_normals.view(b,h*w,3), vertice_normals_center.view(b,(h-1)*(w-1),3)], 1)  # Bx(H*W)x2, [0,1]

    idx_map = torch.arange(h*w).reshape(h,w)
    idx_map_center = torch.arange((h-1)*(w-1)).reshape(h-1,w-1)
    faces1 = torch.stack([idx_map[:h-1,:w-1], idx_map[1:,:w-1], idx_map_center+h*w], -1).reshape(-1,3).repeat(b,1,1).int()  # Bx((H-1)*(W-1))x4
    faces2 = torch.stack([idx_map[1:,:w-1], idx_map[1:,1:], idx_map_center+h*w], -1).reshape(-1,3).repeat(b,1,1).int()  # Bx((H-1)*(W-1))x4
    faces3 = torch.stack([idx_map[1:,1:], idx_map[:h-1,1:], idx_map_center+h*w], -1).reshape(-1,3).repeat(b,1,1).int()  # Bx((H-1)*(W-1))x4
    faces4 = torch.stack([idx_map[:h-1,1:], idx_map[:h-1,:w-1], idx_map_center+h*w], -1).reshape(-1,3).repeat(b,1,1).int()  # Bx((H-1)*(W-1))x4
    faces = torch.cat([faces1, faces2, faces3, faces4], 1)

    objs = []
    mtls = []
    for bi in range(b):
        obj = "# OBJ File:"
        obj += "\n\nmtllib $MTLFILE"
        obj += "\n\n# vertices:"
        for v in vertices[bi]:
            obj += "\nv " + " ".join(["%.4f"%x for x in v])
        obj += "\n\n# vertice textures:"
        for vt in vertice_textures[bi]:
            obj += "\nvt " + " ".join(["%.4f"%x for x in vt])
        obj += "\n\n# vertice normals:"
        for vn in vertice_normals[bi]:
            obj += "\nvn " + " ".join(["%.4f"%x for x in vn])
        obj += "\n\n# faces:"
        obj += "\n\nusemtl tex"
        for f in faces[bi]:
            obj += "\nf " + " ".join(["%d/%d/%d"%(x+1,x+1,x+1) for x in f])
        objs += [obj]

        mtl = "newmtl tex"
        mtl += "\nKa 1.0000 1.0000 1.0000"
        mtl += "\nKd 1.0000 1.0000 1.0000"
        mtl += "\nKs 0.0000 0.0000 0.0000"
        mtl += "\nd 1.0"
        mtl += "\nillum 0"
        mtl += "\nmap_Kd $TXTFILE"
        mtls += [mtl]
    return objs, mtls


def depth_to_3d_grid(depth, inv_K=None):
    if inv_K is None:
        image_size = 128
        fov = 12
        R = [[[1.,0.,0.],
              [0.,1.,0.],
              [0.,0.,1.]]]
        R = torch.FloatTensor(R).cuda()
        t = torch.zeros(1,3, dtype=torch.float32).cuda()
        fx = (image_size-1)/2/(math.tan(fov/2 *math.pi/180))
        fy = (image_size-1)/2/(math.tan(fov/2 *math.pi/180))
        cx = (image_size-1)/2
        cy = (image_size-1)/2
        K = [[fx, 0., cx],
             [0., fy, cy],
             [0., 0., 1.]]
        K = torch.FloatTensor(K).cuda()
        inv_K = torch.inverse(K).unsqueeze(0)
    b, h, w = depth.shape
    grid_2d = get_grid(b, h, w, normalize=False).to(depth.device)  # Nxhxwx2
    depth = depth.unsqueeze(-1)
    grid_3d = torch.cat((grid_2d, torch.ones_like(depth)), dim=3)
    grid_3d = grid_3d.matmul(inv_K.transpose(2,1)) * depth
    return grid_3d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_G', type=str)
    parser.add_argument('path_D', type=str)
    parser.add_argument('--curriculum', type=str, default='CelebA_ShadeGAN_noview')
    parser.add_argument('--output_dir', type=str, default='imgs')
    parser.add_argument('--img_list', type=str, default='./inversion/list.txt')
    parser.add_argument('--root', type=str, default='./inversion')
    parser.add_argument('--max_batch_size', type=int, default=2400000)
    parser.add_argument('--lock_view_dependence', action='store_true')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--reg_lambda', type=float, default=1e-3)
    parser.add_argument('--iterations', nargs='+', type=int, default=[150, 250, 300])
    parser.add_argument('--psi', type=float, default=1.)
    parser.add_argument('--delta', type=float, default=0.0325)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--num_steps', type=int, default=12)
    opt = parser.parse_args()
    
    os.makedirs(opt.output_dir, exist_ok=True)

    curriculum = getattr(curriculums, opt.curriculum)
    metadata = curriculums.extract_metadata(curriculum, 0)
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

    options_dict = {
        'num_steps':6,
        'img_size':128,
        'hierarchical_sample':True,
        'psi':opt.psi,
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
    copied_options_dict = copy.deepcopy(options_dict)
    copied_options_dict['num_steps'] = opt.num_steps
    
    torch.manual_seed(0)
    
    img_list = open(opt.img_list, 'r')

    with torch.no_grad():
        z = torch.randn((10000, 256), device=device)
        frequencies, phase_shifts = generator.siren.mapping_network(z)
        freq_mean = frequencies.mean(0, keepdim=True)
        phase_mean = phase_shifts.mean(0, keepdim=True)

    for i, line in enumerate(img_list.readlines()):
        im_name = line.split()[0]
        im_path = os.path.join(opt.root, im_name)
        img = load_image(im_path, image_size=128)  # use crop=True for cropping celeba aligned images
        im_name = im_name.split('.')[0]
        
        with torch.no_grad():
            _, latent, pose_pred, l, freq, phase = encoder(img, 1, encode=True, **options_dict)
            l = l.reshape(1, 4).detach()

        w_freq = freq.clone()
        w_phase = phase.clone()
        pose = pose_pred.clone()
        w_freq.requires_grad = True
        w_phase.requires_grad = True
        pose.requires_grad = True
        l.requires_grad = True
        optimizer = torch.optim.Adam(
            [{'params': w_freq, 'lr': opt.lr},
             {'params': w_phase, 'lr': opt.lr},
             {'params': pose, 'lr': opt.lr},
             {'params': l, 'lr': opt.lr}], lr=opt.lr)

        for j in range(opt.iterations[-1]):
            if j in opt.iterations:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
            w_freq_n, w_phase_n = w_freq, w_phase
            img_recon = generator.forward_with_frequencies(latent, w_freq_n, w_phase_n, l, pose=pose, **options_dict)['rgb']
            
            loss_mse = F.mse_loss(img_recon, img)
            loss_reg = opt.reg_lambda * ((w_freq_n - freq_mean) ** 2).sum() + opt.reg_lambda * ((w_phase_n - phase_mean) ** 2).sum()
            loss = loss_mse + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if j % 10 == 0:
                print(f'{j}:  MSE loss: {loss_mse.item():.5f}  Regularization loss: {loss_reg.item():.5f}')

        with torch.no_grad():
            results = generator.staged_forward_with_frequencies(latent, w_freq_n, w_phase_n, l, pose=pose, rt_normal=True, max_batch_size=10000, **copied_options_dict)
            img_recon, depth_map, normal_map = results['rgb'], results['depth'], results['normal']
            normal_map = rotate_normal(normal_map, pose, device)
        
        torch.save(torch.cat([w_freq_n, w_phase_n], dim=-1), os.path.join(opt.output_dir, f"img_{im_name}_latent.pth"))
        torch.save(l, os.path.join(opt.output_dir, f"img_{im_name}_light.pth"))
        torch.save(pose, os.path.join(opt.output_dir, f"img_{im_name}_pose.pth"))
        save_image(img_recon, os.path.join(opt.output_dir, f"img_recon_{im_name}.png"), normalize=True)
        save_image(depth_map.unsqueeze(1), os.path.join(opt.output_dir, f"depth_recon_{im_name}.png"), normalize=True)
        vertices = depth_to_3d_grid(depth_map)  # BxHxWx3
        objs, mtls = export_to_obj_string(vertices, normal_map.permute(0,2,3,1))
        with open(os.path.join(opt.output_dir, f'./{im_name}.mtl'), "w") as f:
            f.write(mtls[0].replace('$TXTFILE', f'./img_recon_{im_name}.png'))
        with open(os.path.join(opt.output_dir, f'{im_name}.obj'), "w") as f:
            f.write(objs[0].replace('$MTLFILE', f'./{im_name}.mtl'))

        save_image(normal_map/2+0.5, os.path.join(opt.output_dir, f"normal_render_{im_name}.png"), normalize=False)
        if metadata['shading']:
            save_image(results['shading'].unsqueeze(1), os.path.join(opt.output_dir, f"shading_{im_name}.png"), normalize=True)
