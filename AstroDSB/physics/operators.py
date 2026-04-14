import torch
import torch.nn.functional as F

def dx(u):
    # Central difference: more accurate, symmetric gradients
    # Pad both sides so output shape matches input
    u_pad = F.pad(u, (1, 1, 0, 0), mode='replicate')   # or 'replicate'
    return (u_pad[:, :, :, 2:] - u_pad[:, :, :, :-2]) * 0.5

def dy(u):
    u_pad = F.pad(u, (0, 0, 1, 1), mode='replicate')
    return (u_pad[:, :, 2:, :] - u_pad[:, :, :-2, :]) * 0.5

def divergence(Bx, By):
    # No extra pad needed — central diff preserves shape
    return dx(Bx) + dy(By)

def polarization_angle(Bx, By):
    """
    ψ = 0.5 * arctan2(2·Bx·By,  Bx² - By²)
    arctan2 handles the quadrant correctly and avoids division-by-zero
    at Bx == By.
    """
    numerator   = 2.0 * Bx * By
    denominator = Bx ** 2 - By ** 2
    return 0.5 * torch.atan2(numerator, denominator)