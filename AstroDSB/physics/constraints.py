import torch
import torch.nn.functional as F

def gradient(u):
    dx = u[:, :, :, 1:] - u[:, :, :, :-1]
    dy = u[:, :, 1:, :] - u[:, :, :-1, :]
    dx = F.pad(dx, (0,1,0,0))
    dy = F.pad(dy, (0,0,0,1))
    return dx, dy


def smoothness_loss(rho):
    dx, dy = gradient(rho)
    return (dx.abs() + dy.abs()).mean()


def mass_conservation_loss(rho):
    total_mass = rho.sum(dim=[2,3])
    return ((total_mass - total_mass.mean())**2).mean()