import math

import torch
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal as MVN


class LSampler():
    def __init__(self, device, dataset='CelebA', mvn_path=None):
        self.device = device
        self.dataset = dataset

        data2name = {'CelebA': 'celeba', 'Synface': 'synface', 'Cats': 'cat'}
        name = data2name[dataset]

        if mvn_path is None:
            mvn_path = f'weights/light_mvn/{name}_light_mvn.pth'
        self.light_mvn = torch.load(mvn_path)

        self.ldist = MVN(self.light_mvn['mean'].to(device), self.light_mvn['cov'].to(device))

    def sample(self, batchsize):
        with torch.no_grad():
            output = self.ldist.sample((batchsize,))
            return output


def z_sampler(shape, device, dist='gaussian'):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z


def act_scheduler(iteration, start=5000, end=7000):
    factor = (iteration - start) / (end - start)
    return min(max(factor, 0), 1)


def rendering_scheduler(iteration, beta=5.5e-5, start_step=5000, delta_max=0.24, delta_min=0.06, num_steps_max=12, num_steps_min=3, **kwargs):
    if iteration <= start_step:
        delta = -1
        num = num_steps_max
    else:
        delta = (delta_max - delta_min) * math.exp(-(iteration - start_step) * beta) + delta_min
        num = (num_steps_max - num_steps_min) * math.exp(-(iteration - start_step) * beta) + num_steps_min
        num = round(num)
    return delta, num


def resize(input, size):
    dim = input.dim()
    if dim == 3:
        input = input.unsqueeze(1)
    if input.size(-1) > size:
        input = F.interpolate(input, size, mode='area')
    elif input.size(-1) < size:
        input = F.interpolate(input, size, mode='bilinear')
    
    if dim == 3:
        input = input.squeeze(1)
    return input
