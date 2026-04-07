import numpy as np
import torch

from .util import unsqueeze_xdim


def compute_gaussian_product_coef(sigma1, sigma2):
    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var


class Diffusion:
    def __init__(self, betas, device):
        self.device = device

        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        # For q(x_t | y_noisy, x1), x_t should equal y_noisy near t=0 and x1 near t=T.
        # compute_gaussian_product_coef(std_fwd, std_bwd) returns weights for
        # (y_noisy, x1) in that order.
        mu_y, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        to_torch = lambda array: torch.tensor(array, dtype=torch.float32, device=device)
        self.betas = to_torch(betas)
        self.std_fwd = to_torch(std_fwd)
        self.std_bwd = to_torch(std_bwd)
        self.std_sb = to_torch(std_sb)
        self.mu_x1 = to_torch(mu_x1)
        self.mu_y = to_torch(mu_y)

    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)

    def q_sample(self, step, x1, y_noisy, ot_ode=False):
        _batch, *xdim = x1.shape
        mu_x1 = unsqueeze_xdim(self.mu_x1[step], xdim)
        mu_y = unsqueeze_xdim(self.mu_y[step], xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x1 * x1 + mu_y * y_noisy
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
        return xt.detach()

    def endpoint_conditioned_step(self, step, next_step, x_t, pred_x1, y_noisy, ot_ode=False):
        if next_step <= step:
            raise ValueError(
                f"Expected endpoint-conditioned bridge step with next_step > step, got step={step}, next_step={next_step}"
            )
        if x_t.shape != pred_x1.shape or x_t.shape != y_noisy.shape:
            raise ValueError(
                f"Bridge step expects x_t, pred_x1, and y_noisy to share shape; got {x_t.shape}, {pred_x1.shape}, {y_noisy.shape}"
            )

        cur = torch.full(
            (x_t.shape[0],),
            step,
            device=self.device,
            dtype=torch.long,
        )
        nxt = torch.full(
            (x_t.shape[0],),
            next_step,
            device=self.device,
            dtype=torch.long,
        )

        var_cur = self.std_fwd[cur] ** 2
        var_next = self.std_fwd[nxt] ** 2
        total_var = self.std_fwd[-1] ** 2
        delta_var = var_next - var_cur
        remaining_var = total_var - var_cur
        next_remaining_var = total_var - var_next
        if torch.any(delta_var <= 0):
            raise ValueError(f"Expected increasing bridge variance, got {delta_var}")
        if torch.any(remaining_var <= 0):
            raise ValueError(f"Bridge remaining variance must stay positive, got {remaining_var}")

        coeff = unsqueeze_xdim(delta_var / remaining_var, x_t.shape[1:])
        mean = x_t + coeff * (pred_x1 - x_t)
        if ot_ode:
            return mean.detach()

        noise_var = (delta_var * next_remaining_var) / remaining_var
        noise_std = unsqueeze_xdim(noise_var.clamp_min(0.0).sqrt(), x_t.shape[1:])
        x_next = mean + noise_std * torch.randn_like(x_t)
        return x_next.detach()
