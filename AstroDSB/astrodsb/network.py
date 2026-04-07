import torch

from guided_diffusion.script_util import create_model

from . import util


class AstroNet(torch.nn.Module):
    def __init__(
        self,
        log,
        noise_levels,
        *,
        input_channels,
        cond_channels,
        image_size=128,
        model_channels=128,
        num_res_blocks=2,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.cond_channels = cond_channels
        self.noise_levels = noise_levels
        self.diffusion_model = create_model(
            image_size=image_size,
            num_channels=model_channels,
            num_res_blocks=num_res_blocks,
            in_channels=input_channels + cond_channels,
            out_channels=input_channels,
        )
        log.info(f"[Net] Initialized AstroNet! Size={util.count_parameters(self.diffusion_model)}!")

    def forward(self, x, steps, cond):
        if cond is None:
            raise ValueError("AstroDSB requires cond=y for every forward pass")
        if x.shape[1] != self.input_channels:
            raise ValueError(f"Expected xt with {self.input_channels} channels, got {x.shape[1]}")
        if cond.shape[1] != self.cond_channels:
            raise ValueError(f"Expected cond with {self.cond_channels} channels, got {cond.shape[1]}")

        t = self.noise_levels[steps].detach()
        if t.dim() != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(f"Invalid time embedding shape {t.shape} for batch {x.shape[0]}")

        model_input = torch.cat([x, cond], dim=1)
        return self.diffusion_model(model_input, t)
