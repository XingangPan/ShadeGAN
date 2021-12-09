import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group_factor=8, upsample=True):
        super(ResBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = upsample

        self.gn1 = nn.GroupNorm(in_channels // group_factor, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(out_channels // group_factor, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.activation = nn.ReLU(inplace=True)
        if self.in_channels != self.out_channels:
            self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.activation(self.gn1(x))
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode='nearest')
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        h = self.conv1(h)
        h = self.activation(self.gn2(h))
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.conv_sc(x)
        return x + h


class ResDecoder(nn.Module):
    def __init__(self, zdim=256, cout=2, size=128, nf=32, gn_base=8, activation=nn.Sigmoid):
        super(ResDecoder, self).__init__()
        extra = int(np.log2(size) - 6)
        for i in range(extra):
            nf *= 2
        self.linear = nn.Linear(zdim, nf*8)
        ## upsampling
        network = [
            nn.Conv2d(nf*8+2, nf*8, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(nf*2, nf*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
            ResBlock(nf*8, nf*4, upsample=True), # 16
            ResBlock(nf*4, nf*2, upsample=True), # 32
            ResBlock(nf*2, nf, upsample=True)] # 64
        for i in range(extra):
            nf = nf // 2
            network += [ResBlock(nf*2, nf, upsample=True)]
        network += [
            nn.GroupNorm(nf // 4, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input, pose):
        x = self.linear(input)
        x = torch.cat([x, pose], dim=-1)
        return self.network(x[...,None,None])
