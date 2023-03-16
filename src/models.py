import numpy as np
import torchvision
import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import e3nn
from e3nn import o3

from src.so3_utils import s2_healpix_grid, flat_wigner, so3_irreps, s2_irreps


class ResNet(nn.Module):
    def __init__(self,
                 size: int=50,
                 pretrained: bool=False,
                 pool_features: bool=False,
                 **kwargs,
                 ):
        super().__init__()
        weights = 'DEFAULT' if pretrained else None
        full_model = eval(f'resnet.resnet{size}')(weights=weights)

        out_fdim = 512 if size in (18, 34) else 2048

        # remove pool and linear
        modules = list(full_model.children())[:-2]

        if pool_features:
            modules.append(nn.AdaptiveAvgPool2d(1))
            self.output_shape = (out_fdim, 1, 1)
        else:
            self.output_shape = (out_fdim, 7, 7)

        self.layers = nn.Sequential(*modules)

    def forward(self, img):
        return self.layers(img)


class S2Convolution(torch.nn.Module):
    def __init__(self, f_in, f_out, lmax, kernel_grid):
        super().__init__()
        self.register_parameter(
            "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        )  # [f_in, f_out, n_s2_pts]
        self.register_buffer(
            "Y", o3.spherical_harmonics_alpha_beta(range(lmax + 1), *kernel_grid, normalization="component")
        )  # [n_s2_pts, psi]
        self.lin = o3.Linear(s2_irreps(lmax), so3_irreps(lmax), f_in=f_in, f_out=f_out, internal_weights=False)

    def forward(self, x):
        psi = torch.einsum("ni,xyn->xyi", self.Y, self.w) / self.Y.shape[0] ** 0.5
        return self.lin(x, weight=psi)


class SO3Convolution(torch.nn.Module):
    def __init__(self, f_in, f_out, lmax, kernel_grid):
        super().__init__()
        self.register_parameter(
            "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
        )  # [f_in, f_out, n_so3_pts]
        self.register_buffer("D", flat_wigner(lmax, *kernel_grid))  # [n_so3_pts, psi]
        self.lin = o3.Linear(so3_irreps(lmax), so3_irreps(lmax), f_in=f_in, f_out=f_out, internal_weights=False)

    def forward(self, x):
        psi = torch.einsum("ni,xyn->xyi", self.D, self.w) / self.D.shape[0] ** 0.5
        return self.lin(x, weight=psi)


class HarmonicS2Features(nn.Module):
    def __init__(self, sphere_fdim, lmax, f_out=1):
        super().__init__()
        self.fdim = sphere_fdim
        self.lmax = lmax

        # (f_in, f_out, (lmax+1)**2)
        weight = torch.zeros((self.fdim, f_out, (lmax+1)**2), dtype=torch.float32)
        self.weight = nn.Parameter(data=weight, requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.weight)

        # (f_out, f_in)
        bias = torch.zeros((self.fdim, f_out), dtype=torch.float32)
        self.bias = nn.Parameter(data=bias, requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.bias)

    def forward(self):
        return self.weight, self.bias

    def __repr__(self):
        return f'HarmonicS2Features(fdim={self.fdim}, lmax={self.lmax})'


class SpatialS2Features(nn.Module):
    def __init__(self, sphere_fdim, lmax, rec_level=1, f_out=1):
        super().__init__()
        self.fdim = sphere_fdim
        self.lmax = lmax

        alpha, beta = s2_healpix_grid(max_beta=np.inf, rec_level=rec_level)
        self.register_buffer(
            "Y", o3.spherical_harmonics_alpha_beta(range(lmax+1), alpha, beta, normalization='component')
        )

        # (f_in, f_out, (lmax+1)**2)
        weight = torch.zeros((self.fdim, f_out, alpha.shape[0]), dtype=torch.float32)
        self.weight = nn.Parameter(data=weight, requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.weight)

        bias = torch.zeros((self.fdim, f_out), dtype=torch.float32)
        self.bias = nn.Parameter(data=bias, requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.bias)

    def forward(self):
        x = torch.einsum("ni,xyn->xyi", self.Y, self.weight) / self.Y.shape[0]**0.5
        return x.unsqueeze(1), self.bias


class HarmonicS2Projector(nn.Module):
    def __init__(self,
                 input_shape: tuple,
                 sphere_fdim: int,
                 lmax: int,
                ):
        super().__init__()
        self.sphere_fdim = sphere_fdim
        self.n_harmonics = (lmax+1)**2

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.lin1 = nn.Linear(input_shape[0], 512)
        self.lin2 = nn.Linear(512, sphere_fdim*self.n_harmonics)

        self.norm_act = e3nn.nn.S2Activation(s2_irreps(lmax), torch.relu, 20)

    def forward(self, x):
        x = torch.flatten(self.avg_pool(x), 1)
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x).view(x.size(0), self.sphere_fdim, self.n_harmonics)
        x = self.norm_act(x)
        return x


class SpatialS2Projector(nn.Module):
    def __init__(self,
                 input_shape: tuple,
                 sphere_fdim: int,
                 lmax: int,
                 coverage: float=0.9,
                 sigma: float=0.2,
                 n_subset: int=20,
                 max_beta: float=np.radians(90),
                 rec_level: int=2,
                 taper_beta: float=np.radians(75),
                ):
        '''Project from feature map to spherical signal

        ToDo:
            subsample grid points in train mode (might require a buffer)
            add noise to grid location in train mode
        '''
        super().__init__()
        self.n_subset = n_subset

        fmap_size = input_shape[1]

        self.conv1x1 = nn.Conv2d(input_shape[0], sphere_fdim*1, 1)

        # north pole is at y=+1
        self.kernel_grid = s2_healpix_grid(max_beta=max_beta, rec_level=rec_level)

        self.xyz = o3.angles_to_xyz(*self.kernel_grid)

        # orthographic projection
        max_radius = torch.linalg.norm(self.xyz[:,[0,2]], dim=1).max()
        sample_x = coverage * self.xyz[:,2] / max_radius # range -1 to 1
        sample_y = coverage * self.xyz[:,0] / max_radius

        gridx, gridy = torch.meshgrid(2*[torch.linspace(-1,1,fmap_size)], indexing='ij')
        scale = 1 / np.sqrt(2 * np.pi * sigma**2)
        data = scale * torch.exp(-((gridx.unsqueeze(-1) - sample_x).pow(2) \
                                   +(gridy.unsqueeze(-1) - sample_y).pow(2)) / (2*sigma**2) )
        data = data / data.sum((0,1), keepdims=True)

        # apply mask to taper magnitude near border if desired
        betas = self.kernel_grid[1]
        if taper_beta < max_beta:
            mask = ((betas - max_beta)/(taper_beta - max_beta)).clamp(max=1).view(1,1,-1)
        else:
            mask = torch.ones_like(data)

        data = (mask * data).unsqueeze(0).unsqueeze(0).to(torch.float32)
        self.weight = nn.Parameter(data= data, requires_grad=True)

        self.n_pts = self.weight.shape[-1]
        self.ind = torch.arange(self.n_pts)

        self.register_buffer(
            "Y", o3.spherical_harmonics_alpha_beta(range(lmax+1),
                                                   *self.kernel_grid,
                                                   normalization='component')
        )

    def forward(self, x):
        '''
        :x: float tensor of shape (B,C,H,W)
        :return: feature vector of shape (B,P,C) where P is number of points on S2
        '''
        x = self.conv1x1(x)

        if self.n_subset is not None:
            self.ind = torch.randperm(self.n_pts)[:self.n_subset]

        x = (x.unsqueeze(-1) * self.weight[..., self.ind]).sum((2,3))
        x = torch.relu(x)
        x = torch.einsum('ni,xyn->xyi', self.Y[self.ind], x) / self.ind.shape[0]**0.5
        return x
