"""
To easily reproduce experiments, and avoid passing several command line arguments, we implemented
a curriculum utility. Parameters can be set in a curriculum dictionary.

Curriculum Schema:

    Numerical keys in the curriculum specify an upsample step. When the current step matches the upsample step,
    the values in the corresponding dict be updated in the curriculum. Common curriculum values specified at upsamples:
        batch_size: Batch Size.
        num_steps: Number of samples along ray.
        img_size: Generated image resolution.
        batch_split: Integer number over which to divide batches and aggregate sequentially. (Used due to memory constraints)
        gen_lr: Generator learnig rate.
        disc_lr: Discriminator learning rate.

    fov: Camera field of view
    ray_start: Near clipping for camera rays.
    ray_end: Far clipping for camera rays.
    fade_steps: Number of steps to fade in new layer on discriminator after upsample.
    h_stddev: Stddev of camera yaw in radians.
    v_stddev: Stddev of camera pitch in radians.
    h_mean:  Mean of camera yaw in radians.
    v_mean: Mean of camera yaw in radians.
    sample_dist: Type of camera pose distribution. (gaussian | spherical_uniform | uniform)
    topk_interval: Interval over which to fade the top k ratio.
    topk_v: Minimum fraction of a batch to keep during top k training.
    betas: Beta parameters for Adam.
    unique_lr: Whether to use reduced LRs for mapping network.
    weight_decay: Weight decay parameter.
    r1_lambda: R1 regularization parameter.
    latent_dim: Latent dim for Siren network  in generator.
    grad_clip: Grad clipping parameter.
    model: Siren architecture used in generator. (SPATIALSIRENBASELINE | TALLSIREN)
    generator: Generator class. (ImplicitGenerator3d)
    discriminator: Discriminator class. (ProgressiveEncoderDiscriminator | ProgressiveDiscriminator)
    dataset: Training dataset. (CelebA | Carla | Cats)
    clamp_mode: Clamping function for Siren density output. (relu | softplus)
    z_dist: Latent vector distributiion. (gaussian | uniform)
    hierarchical_sample: Flag to enable hierarchical_sampling from NeRF algorithm. (Doubles the number of sampled points)
    z_labmda: Weight for experimental latent code positional consistency loss.
    pos_lambda: Weight parameter for experimental positional consistency loss.
    last_back: Flag to fill in background color with last sampled color on ray.
    shading: Whether to model shading.
    view_condition: Whether color is conditioned on camera view
    light_condition: Whether color is conditioned on lighting
    surf_track: Whether to use surface tracking network
    start_step: The iteration step to start using surface tracking network for rendering
"""

import math

def next_upsample_step(curriculum, current_step):
    # Return the epoch when it will next upsample
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata['img_size']
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step > current_step and curriculum[curriculum_step].get('img_size', 512) > current_size:
            return curriculum_step
    return float('Inf')

def last_upsample_step(curriculum, current_step):
    # Returns the start epoch of the current stage, i.e. the epoch
    # it last upsampled
    current_metadata = extract_metadata(curriculum, current_step)
    current_size = current_metadata['img_size']
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int]):
        if curriculum_step <= current_step and curriculum[curriculum_step]['img_size'] == current_size:
            return curriculum_step
    return 0

def get_current_step(curriculum, epoch):
    step = 0
    for update_epoch in curriculum['update_epochs']:
        if epoch >= update_epoch:
            step += 1
    return step

def extract_metadata(curriculum, current_step):
    return_dict = {}
    for curriculum_step in sorted([cs for cs in curriculum.keys() if type(cs) == int], reverse=True):
        if curriculum_step <= current_step:
            for key, value in curriculum[curriculum_step].items():
                return_dict[key] = value
            break
    for key in [k for k in curriculum.keys() if type(k) != int]:
        return_dict[key] = curriculum[key]
    return return_dict


CelebA_piGAN = {
    0: {'batch_size': 13*2, 'num_steps': 12, 'img_size': 32, 'batch_split': 1, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    int(30e3): {'batch_size': 8*2, 'num_steps': 5, 'img_size': 64, 'batch_split': 2, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    int(70e3): {'batch_size': 8*2, 'num_steps': 5, 'img_size': 64, 'batch_split': 2, 'gen_lr': 1e-5, 'disc_lr': 1e-4},
    int(100e3): {'batch_size': 2*2, 'num_steps': 4, 'img_size': 128, 'batch_split': 2, 'gen_lr': 2e-6, 'disc_lr': 2e-5},
    int(150e3): {},

    'dataset_path': '/path/to/dataset/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 1,
    'latent_dim': 256,
    'grad_clip': 0.3,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'ProgressiveEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'shading': False,
    'view_condition': True,
    'light_condition': False,
    'surf_track': False,
}

CelebA_ShadeGAN_noview = {
    0: {'batch_size': 13*2, 'num_steps': 12, 'img_size': 32, 'batch_split': 1, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    int(30e3): {'batch_size': 8*2, 'num_steps': 5, 'img_size': 64, 'batch_split': 2, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    int(70e3): {'batch_size': 8*2, 'num_steps': 5, 'img_size': 64, 'batch_split': 2, 'gen_lr': 1e-5, 'disc_lr': 1e-4},
    int(100e3): {'batch_size': 2*2, 'num_steps': 4, 'img_size': 128, 'batch_split': 2, 'gen_lr': 2e-6, 'disc_lr': 2e-5},
    int(150e3): {},

    'dataset_path': '/path/to/dataset/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': True,
    'weight_decay': 0,
    'r1_lambda': 1,
    'latent_dim': 256,
    'grad_clip': 0.3,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'ProgressiveEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15.,
    'last_back': False,
    'shading': True,
    'view_condition': False,
    'light_condition': True,
    'surf_track': True,
    'start_step': 5000,
    'beta': 3e-5,
    'delta_max': 0.24,
    'delta_min': 0.06,
    'num_steps_max': 12,
    'num_steps_min': 6,
}

CelebA_ShadeGAN_view = {
    0: {'batch_size': 13*2, 'num_steps': 12, 'img_size': 32, 'batch_split': 1, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    int(30e3): {'batch_size': 8*2, 'num_steps': 5, 'img_size': 64, 'batch_split': 2, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    int(70e3): {'batch_size': 8*2, 'num_steps': 5, 'img_size': 64, 'batch_split': 2, 'gen_lr': 1e-5, 'disc_lr': 1e-4},
    int(100e3): {'batch_size': 2*2, 'num_steps': 4, 'img_size': 128, 'batch_split': 2, 'gen_lr': 2e-6, 'disc_lr': 2e-5},
    int(150e3): {},

    'dataset_path': '/path/to/dataset/*.jpg',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': True,
    'weight_decay': 0,
    'r1_lambda': 1,
    'latent_dim': 256,
    'grad_clip': 0.3,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'ProgressiveEncoderDiscriminator',
    'dataset': 'CelebA',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15.,
    'last_back': False,
    'shading': True,
    'view_condition': True,
    'light_condition': True,
    'surf_track': True,
    'start_step': 5000,
    'beta': 3e-5,
    'delta_max': 0.24,
    'delta_min': 0.06,
    'num_steps_max': 12,
    'num_steps_min': 6,
}

CATS_piGAN = {
    0: {'batch_size': 32, 'num_steps': 20, 'img_size': 64, 'batch_split': 4, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(30e3): {},

    'dataset_path': '/path/to/dataset/*.jpg',
    'fov': 12,
    'ray_start': 0.8,
    'ray_end': 1.2,
    'fade_steps': 10000,
    'h_stddev': 0.5,
    'v_stddev': 0.4,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'uniform',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256,
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'StridedDiscriminator',
    'dataset': 'Cats',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 0,
    'last_back': False,
    'eval_last_back': True,
    'shading': False,
    'view_condition': True,
    'light_condition': False,
    'surf_track': False,
}

CATS_ShadeGAN = {
    0: {'batch_size': 32, 'num_steps': 20, 'img_size': 64, 'batch_split': 8, 'gen_lr': 6e-5, 'disc_lr': 2e-4},
    int(30e3): {},

    'dataset_path': '/path/to/dataset/*.jpg',
    'fov': 12,
    'ray_start': 0.8,
    'ray_end': 1.2,
    'fade_steps': 10000,
    'h_stddev': 0.5,
    'v_stddev': 0.4,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'uniform',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 0.2,
    'latent_dim': 256,
    'grad_clip': 10,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'StridedDiscriminator',
    'dataset': 'Cats',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 0,
    'last_back': False,
    'eval_last_back': True,
    'shading': True,
    'view_condition': True,
    'light_condition': True,
    'surf_track': True,
    'start_step': 5000,
    'beta': 1e-4,
    'delta_max': 0.24,
    'delta_min': 0.06,
    'num_steps_max': 20,
    'num_steps_min': 10,
}

BFM_piGAN = {
    0: {'batch_size': 24, 'num_steps': 12, 'img_size': 32, 'batch_split': 1, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    int(20e3): {'batch_size': 12, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    int(40e3): {'batch_size': 12, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'gen_lr': 1e-5, 'disc_lr': 1e-4},
    int(50e3): {'batch_size': 4, 'num_steps': 12, 'img_size': 128, 'batch_split': 2, 'gen_lr': 2e-6, 'disc_lr': 2e-5},
    int(81e3): {},

    'dataset_path': '/path/to/dataset/*.png',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': False,
    'weight_decay': 0,
    'r1_lambda': 1,
    'latent_dim': 256,
    'grad_clip': 0.3,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'ProgressiveEncoderDiscriminator',
    'dataset': 'Synface',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15,
    'last_back': False,
    'eval_last_back': True,
    'shading': False,
    'view_condition': True,
    'light_condition': False,
    'surf_track': False,
}

BFM_ShadeGAN_noview = {
    0: {'batch_size': 24, 'num_steps': 12, 'img_size': 32, 'batch_split': 1, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    int(20e3): {'batch_size': 12, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    int(40e3): {'batch_size': 12, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'gen_lr': 1e-5, 'disc_lr': 1e-4},
    int(50e3): {'batch_size': 4, 'num_steps': 12, 'img_size': 128, 'batch_split': 2, 'gen_lr': 2e-6, 'disc_lr': 2e-5},
    int(81e3): {},

    'dataset_path': '/path/to/dataset/*.png',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': True,
    'weight_decay': 0,
    'r1_lambda': 1,
    'latent_dim': 256,
    'grad_clip': 0.3,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'ProgressiveEncoderDiscriminator',
    'dataset': 'Synface',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15.,
    'last_back': False,
    'shading': True,
    'neural_light': False,
    'view_condition': False,
    'light_condition': True,
    'surf_track': False,
    'delta': -1,
}

BFM_ShadeGAN_view = {
    0: {'batch_size': 24, 'num_steps': 12, 'img_size': 32, 'batch_split': 1, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    int(20e3): {'batch_size': 12, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'gen_lr': 2e-5, 'disc_lr': 2e-4},
    int(40e3): {'batch_size': 12, 'num_steps': 12, 'img_size': 64, 'batch_split': 2, 'gen_lr': 1e-5, 'disc_lr': 1e-4},
    int(50e3): {'batch_size': 4, 'num_steps': 12, 'img_size': 128, 'batch_split': 2, 'gen_lr': 2e-6, 'disc_lr': 2e-5},
    int(81e3): {},

    'dataset_path': '/path/to/dataset/*.png',
    'fov': 12,
    'ray_start': 0.88,
    'ray_end': 1.12,
    'fade_steps': 10000,
    'h_stddev': 0.3,
    'v_stddev': 0.155,
    'h_mean': math.pi*0.5,
    'v_mean': math.pi*0.5,
    'sample_dist': 'gaussian',
    'topk_interval': 2000,
    'topk_v': 0.6,
    'betas': (0, 0.9),
    'unique_lr': True,
    'weight_decay': 0,
    'r1_lambda': 1,
    'latent_dim': 256,
    'grad_clip': 0.3,
    'model': 'SPATIALSIRENBASELINE',
    'generator': 'ImplicitGenerator3d',
    'discriminator': 'ProgressiveEncoderDiscriminator',
    'dataset': 'Synface',
    'clamp_mode': 'relu',
    'z_dist': 'gaussian',
    'hierarchical_sample': True,
    'z_lambda': 0,
    'pos_lambda': 15.,
    'last_back': False,
    'shading': True,
    'neural_light': False,
    'view_condition': True,
    'light_condition': True,
    'surf_track': False,
    'delta': -1,
}
