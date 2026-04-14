from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .operators import dx, dy

def edge_aware_smoothness_loss(
    prediction: torch.Tensor,
    observation: torch.Tensor | None = None,
    *,
    edge_scale: float = 8.0,
) -> torch.Tensor:
    """
    Weak edge-aware TV prior.

    When an observation is provided, gradients that align with strong observed edges
    are penalized less aggressively than gradients in flat regions.
    """
    pred_dx = dx(prediction)
    pred_dy = dy(prediction)

    if observation is None:
        weight_x = 1.0
        weight_y = 1.0
    else:
        obs_dx = dx(observation).detach().abs()
        obs_dy = dy(observation).detach().abs()
        weight_x = torch.exp(-edge_scale * obs_dx)
        weight_y = torch.exp(-edge_scale * obs_dy)

    return (weight_x * pred_dx.abs() + weight_y * pred_dy.abs()).mean()


def normalized_range_loss(
    prediction: torch.Tensor,
    *,
    lower: float = -1.0,
    upper: float = 1.0,
) -> torch.Tensor:
    """
    Penalize predictions that leave the normalized training support.
    """
    below = F.relu(lower - prediction)
    above = F.relu(prediction - upper)
    return (below.pow(2) + above.pow(2)).mean()


def identity_observation_consistency_loss(
    prediction: torch.Tensor,
    observation: torch.Tensor,
) -> torch.Tensor:
    """
    Optional placeholder consistency term for tasks where the observation and target
    live in the same scalar image space.

    This is intentionally not enabled by default for the density task because the
    paper's density observation is a projected quantity rather than the target itself.
    """
    if prediction.shape != observation.shape:
        raise ValueError(
            f"Identity observation consistency requires matching shapes, got {prediction.shape} and {observation.shape}"
        )
    return F.mse_loss(prediction, observation)


def _normalized_to_physical(
    tensor: torch.Tensor,
    *,
    min_value: float,
    max_value: float,
) -> torch.Tensor:
    scale = float(max_value) - float(min_value)
    if scale <= 0.0:
        return tensor.new_full(tensor.shape, float(min_value))
    return tensor.add(1.0).mul(0.5).mul(scale).add(float(min_value))


def _physical_to_normalized(
    tensor: torch.Tensor,
    *,
    min_value: float,
    max_value: float,
) -> torch.Tensor:
    scale = float(max_value) - float(min_value)
    if scale <= 0.0:
        return torch.zeros_like(tensor)
    return tensor.sub(float(min_value)).div(scale).mul(2.0).sub(1.0)


class SlabColumnDensityOperator:
    """
    Differentiable forward model for the current 2D density pipeline.

    Assumption:
    - target x1 is a log-density map, e.g. log10(n_H)
    - observation y is a log-column-density map, e.g. log10(N_H)
    - a local slab approximation gives log10(N_H) = log10(n_H) + log10(L_eff)

    The operator works in denormalized physical space, then maps back to the
    normalized observation range used by the network.
    """

    def __init__(
        self,
        *,
        target_min: float,
        target_max: float,
        observation_min: float,
        observation_max: float,
        log_effective_depth: float = 0.0,
    ) -> None:
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        self.observation_min = float(observation_min)
        self.observation_max = float(observation_max)
        self.log_effective_depth = float(log_effective_depth)

    def __call__(self, prediction: torch.Tensor) -> torch.Tensor:
        target_physical = _normalized_to_physical(
            prediction,
            min_value=self.target_min,
            max_value=self.target_max,
        )
        projected_physical = target_physical + self.log_effective_depth
        return _physical_to_normalized(
            projected_physical,
            min_value=self.observation_min,
            max_value=self.observation_max,
        )


def observation_consistency_loss(
    prediction: torch.Tensor,
    observation: torch.Tensor,
    *,
    operator,
    noise_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_observation = operator(prediction)
    if predicted_observation.shape != observation.shape:
        raise ValueError(
            f"Forward operator returned shape {predicted_observation.shape}, expected {observation.shape}"
        )
    residual = (predicted_observation - observation) / max(float(noise_scale), 1e-6)
    return residual.pow(2).mean(), predicted_observation


def gradient_consistency_loss(
    predicted_observation: torch.Tensor,
    observation: torch.Tensor,
    *,
    noise_scale: float = 1.0,
) -> torch.Tensor:
    grad_scale = max(float(noise_scale), 1e-6)
    return (
        ((dx(predicted_observation) - dx(observation)) / grad_scale).pow(2).mean()
        + ((dy(predicted_observation) - dy(observation)) / grad_scale).pow(2).mean()
    )


def timestep_weight(
    step: torch.Tensor | None,
    *,
    total_steps: int,
    power: float,
    reference: torch.Tensor,
) -> torch.Tensor:
    if step is None:
        return reference.new_tensor(1.0)
    if total_steps <= 1:
        return reference.new_tensor(1.0)
    normalized = step.float() / float(total_steps - 1)
    weight = normalized.pow(power).mean()
    return weight.to(device=reference.device, dtype=reference.dtype)


@dataclass
class DensityPhysicsLog:
    weight: float
    smooth: float
    in_range: float
    obs_consistency: float
    grad_consistency: float


@dataclass
class MagneticPhysicsLog:
    weight: float
    smooth: float
    in_range: float
    obs_consistency: float
    grad_consistency: float
    dcf_consistency: float


class DensityPhysicsRegularizer:
    def __init__(
        self,
        *,
        w_smooth: float = 0.0,
        w_in_range: float = 0.0,
        w_obs_consistency: float = 0.0,
        w_grad_consistency: float = 0.0,
        observation_operator: str = "none",
        observation_forward_model=None,
        target_stats: dict[str, float] | None = None,
        observation_stats: dict[str, float] | None = None,
        obs_noise_scale: float = 1.0,
        time_weight_power: float = 1.0,
        total_steps: int = 1000,
        edge_scale: float = 8.0,
    ) -> None:
        self.w_smooth = float(w_smooth)
        self.w_in_range = float(w_in_range)
        self.w_obs_consistency = float(w_obs_consistency)
        self.w_grad_consistency = float(w_grad_consistency)
        self.observation_operator = observation_operator
        self.observation_forward_model = observation_forward_model
        self.target_stats = target_stats
        self.observation_stats = observation_stats
        self.obs_noise_scale = float(obs_noise_scale)
        self.time_weight_power = float(time_weight_power)
        self.total_steps = int(total_steps)
        self.edge_scale = float(edge_scale)

    def __call__(
        self,
        prediction: torch.Tensor,
        observation: torch.Tensor,
        *,
        step: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        physical_prediction = prediction
        physical_observation = observation
        if self.target_stats is not None:
            physical_prediction = _normalized_to_physical(
                prediction,
                min_value=self.target_stats["min_value"],
                max_value=self.target_stats["max_value"],
            )
        if self.observation_stats is not None:
            physical_observation = _normalized_to_physical(
                observation,
                min_value=self.observation_stats["min_value"],
                max_value=self.observation_stats["max_value"],
            )

        smooth = edge_aware_smoothness_loss(
            physical_prediction,
            physical_observation,
            edge_scale=self.edge_scale,
        )
        in_range = normalized_range_loss(prediction)
        grad_consistency = prediction.new_zeros(())

        if self.observation_operator == "none" or self.w_obs_consistency <= 0.0:
            obs_consistency = prediction.new_zeros(())
        elif self.observation_operator == "identity":
            obs_consistency = identity_observation_consistency_loss(prediction, observation)
        elif self.observation_operator == "slab_column_density":
            if self.observation_forward_model is None:
                raise ValueError("Missing slab column density forward operator")
            obs_consistency, predicted_observation = observation_consistency_loss(
                prediction,
                observation,
                operator=self.observation_forward_model,
                noise_scale=self.obs_noise_scale,
            )
            if self.w_grad_consistency > 0.0:
                grad_consistency = gradient_consistency_loss(
                    predicted_observation,
                    observation,
                    noise_scale=self.obs_noise_scale,
                )
        else:
            raise ValueError(f"Unsupported density observation operator: {self.observation_operator}")

        base_total = (
            self.w_smooth * smooth
            + self.w_in_range * in_range
            + self.w_obs_consistency * obs_consistency
            + self.w_grad_consistency * grad_consistency
        )
        weight = timestep_weight(
            step,
            total_steps=self.total_steps,
            power=self.time_weight_power,
            reference=prediction,
        )
        total = weight * base_total
        log = DensityPhysicsLog(
            weight=float(weight.detach().item()),
            smooth=float(smooth.detach().item()),
            in_range=float(in_range.detach().item()),
            obs_consistency=float(obs_consistency.detach().item()),
            grad_consistency=float(grad_consistency.detach().item()),
        )
        return total, {
            "weight": log.weight,
            "smooth": log.smooth,
            "in_range": log.in_range,
            "obs_consistency": log.obs_consistency,
            "grad_consistency": log.grad_consistency,
        }


class ChannelProjectionOperator:
    """
    Extract a scalar observed channel for direct consistency with a scalar prediction.
    """

    def __init__(self, *, channel_index: int) -> None:
        self.channel_index = int(channel_index)

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 4:
            raise ValueError(f"Expected observation with shape [N,C,H,W], got {observation.shape}")
        if self.channel_index < 0 or self.channel_index >= observation.shape[1]:
            raise ValueError(
                f"Observation channel index {self.channel_index} is out of range for shape {observation.shape}"
            )
        return observation[:, self.channel_index : self.channel_index + 1]


class DCFMagneticOperator:
    """
    Approximate scalar magnetic-field proxy from observed channels using a DCF-style relation.

    In normalized physical space:
      B_proxy = Q * sqrt(column_density_like) * sigma_v / sigma_phi

    This is only a weak scalar consistency prior for the current pipeline and should
    not be interpreted as a full vector-field MHD constraint.
    """

    def __init__(
        self,
        *,
        column_density_index: int,
        angle_dispersion_index: int,
        velocity_dispersion_index: int,
        observation_stats: dict[str, float],
        target_stats: dict[str, float],
        q_factor: float = 1.0,
        eps: float = 1e-3,
    ) -> None:
        self.column_density_index = int(column_density_index)
        self.angle_dispersion_index = int(angle_dispersion_index)
        self.velocity_dispersion_index = int(velocity_dispersion_index)
        self.observation_stats = observation_stats
        self.target_stats = target_stats
        self.q_factor = float(q_factor)
        self.eps = float(eps)

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 4:
            raise ValueError(f"Expected observation with shape [N,C,H,W], got {observation.shape}")
        max_index = max(
            self.column_density_index,
            self.angle_dispersion_index,
            self.velocity_dispersion_index,
        )
        if max_index >= observation.shape[1]:
            raise ValueError(
                f"DCF operator needs channels up to {max_index}, got observation shape {observation.shape}"
            )

        obs_min = self.observation_stats["min_value"]
        obs_max = self.observation_stats["max_value"]
        target_min = self.target_stats["min_value"]
        target_max = self.target_stats["max_value"]

        column_density = _normalized_to_physical(
            observation[:, self.column_density_index : self.column_density_index + 1],
            min_value=obs_min,
            max_value=obs_max,
        )
        angle_dispersion = _normalized_to_physical(
            observation[:, self.angle_dispersion_index : self.angle_dispersion_index + 1],
            min_value=obs_min,
            max_value=obs_max,
        )
        velocity_dispersion = _normalized_to_physical(
            observation[:, self.velocity_dispersion_index : self.velocity_dispersion_index + 1],
            min_value=obs_min,
            max_value=obs_max,
        )

        shifted_column_density = (column_density - column_density.amin(dim=(-2, -1), keepdim=True)) + self.eps
        shifted_velocity_dispersion = (
            velocity_dispersion - velocity_dispersion.amin(dim=(-2, -1), keepdim=True)
        ) + self.eps
        shifted_angle_dispersion = (
            angle_dispersion - angle_dispersion.amin(dim=(-2, -1), keepdim=True)
        ) + self.eps

        proxy_physical = (
            self.q_factor
            * shifted_column_density.sqrt()
            * shifted_velocity_dispersion
            / shifted_angle_dispersion
        )
        return _physical_to_normalized(
            proxy_physical,
            min_value=target_min,
            max_value=target_max,
        )


class MagneticPhysicsRegularizer:
    def __init__(
        self,
        *,
        w_smooth: float = 0.0,
        w_in_range: float = 0.0,
        w_obs_consistency: float = 0.0,
        w_grad_consistency: float = 0.0,
        w_dcf_consistency: float = 0.0,
        observation_operator=None,
        dcf_operator=None,
        target_stats: dict[str, float] | None = None,
        observation_stats: dict[str, float] | None = None,
        obs_noise_scale: float = 1.0,
        time_weight_power: float = 1.0,
        total_steps: int = 1000,
        edge_scale: float = 8.0,
    ) -> None:
        self.w_smooth = float(w_smooth)
        self.w_in_range = float(w_in_range)
        self.w_obs_consistency = float(w_obs_consistency)
        self.w_grad_consistency = float(w_grad_consistency)
        self.w_dcf_consistency = float(w_dcf_consistency)
        self.observation_operator = observation_operator
        self.dcf_operator = dcf_operator
        self.target_stats = target_stats
        self.observation_stats = observation_stats
        self.obs_noise_scale = float(obs_noise_scale)
        self.time_weight_power = float(time_weight_power)
        self.total_steps = int(total_steps)
        self.edge_scale = float(edge_scale)

    def __call__(
        self,
        prediction: torch.Tensor,
        observation: torch.Tensor,
        *,
        step: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if prediction.shape[1] != 1:
            raise ValueError(f"Magnetic scalar regularizer expects a single-channel prediction, got {prediction.shape}")

        physical_prediction = prediction
        if self.target_stats is not None:
            physical_prediction = _normalized_to_physical(
                prediction,
                min_value=self.target_stats["min_value"],
                max_value=self.target_stats["max_value"],
            )

        projected_observation = None
        if self.observation_operator is not None:
            projected_observation = self.observation_operator(observation)
        physical_projected_observation = projected_observation
        if projected_observation is not None and self.observation_stats is not None:
            physical_projected_observation = _normalized_to_physical(
                projected_observation,
                min_value=self.observation_stats["min_value"],
                max_value=self.observation_stats["max_value"],
            )

        smooth = edge_aware_smoothness_loss(
            physical_prediction,
            physical_projected_observation,
            edge_scale=self.edge_scale,
        )
        in_range = normalized_range_loss(prediction)

        obs_consistency = prediction.new_zeros(())
        grad_consistency = prediction.new_zeros(())
        dcf_consistency = prediction.new_zeros(())

        if self.w_obs_consistency > 0.0:
            if projected_observation is None:
                raise ValueError("Magnetic observation consistency requires an observation operator")
            obs_consistency = ((prediction - projected_observation) / max(self.obs_noise_scale, 1e-6)).pow(2).mean()
            if self.w_grad_consistency > 0.0:
                grad_consistency = gradient_consistency_loss(
                    prediction,
                    projected_observation,
                    noise_scale=self.obs_noise_scale,
                )

        if self.w_dcf_consistency > 0.0:
            if self.dcf_operator is None:
                raise ValueError("Magnetic DCF consistency requires a DCF operator")
            dcf_proxy = self.dcf_operator(observation)
            dcf_consistency = ((prediction - dcf_proxy) / max(self.obs_noise_scale, 1e-6)).pow(2).mean()

        base_total = (
            self.w_smooth * smooth
            + self.w_in_range * in_range
            + self.w_obs_consistency * obs_consistency
            + self.w_grad_consistency * grad_consistency
            + self.w_dcf_consistency * dcf_consistency
        )
        weight = timestep_weight(
            step,
            total_steps=self.total_steps,
            power=self.time_weight_power,
            reference=prediction,
        )
        total = weight * base_total
        log = MagneticPhysicsLog(
            weight=float(weight.detach().item()),
            smooth=float(smooth.detach().item()),
            in_range=float(in_range.detach().item()),
            obs_consistency=float(obs_consistency.detach().item()),
            grad_consistency=float(grad_consistency.detach().item()),
            dcf_consistency=float(dcf_consistency.detach().item()),
        )
        return total, {
            "weight": log.weight,
            "smooth": log.smooth,
            "in_range": log.in_range,
            "obs_consistency": log.obs_consistency,
            "grad_consistency": log.grad_consistency,
            "dcf_consistency": log.dcf_consistency,
        }


def build_physics_regularizer(opt):
    physics_weight = float(getattr(opt, "physics_weight", 0.0))
    if physics_weight <= 0.0:
        return None

    task = getattr(opt, "task", "density")
    if task == "density":
        observation_operator = getattr(opt, "physics_density_observation_operator", "none")
        forward_model = None
        observation_stats = getattr(opt, "observation_normalization", None)
        target_stats = getattr(opt, "target_normalization", None)
        if observation_operator == "slab_column_density":
            if observation_stats is None or target_stats is None:
                raise ValueError(
                    "Density slab column density operator requires observation and target normalization stats"
                )
            forward_model = SlabColumnDensityOperator(
                target_min=target_stats["min_value"],
                target_max=target_stats["max_value"],
                observation_min=observation_stats["min_value"],
                observation_max=observation_stats["max_value"],
                log_effective_depth=getattr(opt, "physics_density_log_effective_depth", 0.0),
            )
        return DensityPhysicsRegularizer(
            w_smooth=getattr(opt, "physics_density_smooth_weight", 0.0),
            w_in_range=getattr(opt, "physics_density_range_weight", 0.0),
            w_obs_consistency=getattr(opt, "physics_density_observation_weight", 0.0),
            w_grad_consistency=getattr(opt, "physics_density_gradient_weight", 0.0),
            observation_operator=observation_operator,
            observation_forward_model=forward_model,
            target_stats=target_stats,
            observation_stats=observation_stats,
            obs_noise_scale=getattr(opt, "obs_noise_scale", 1.0),
            time_weight_power=getattr(opt, "physics_time_weight_power", 1.0),
            total_steps=getattr(opt, "interval", 1000),
            edge_scale=getattr(opt, "physics_density_edge_scale", 8.0),
        )

    if task == "mag":
        observation_stats = getattr(opt, "observation_normalization", None)
        target_stats = getattr(opt, "target_normalization", None)
        if observation_stats is None or target_stats is None:
            raise ValueError("Magnetic physics regularizer requires observation and target normalization stats")

        schema_map = {
            "default_xu2025": {
                "column_density": 0,
                "angle_dispersion": 1,
                "los_velocity_dispersion": 2,
                "projected_b_field": 3,
            }
        }
        schema = getattr(opt, "mag_channel_schema", "default_xu2025")
        if schema not in schema_map:
            raise ValueError(f"Unsupported magnetic channel schema for physics regularizer: {schema}")
        channel_map = schema_map[schema]
        bridge_mode = getattr(opt, "mag_bridge_mode", "projected_b_field")
        if bridge_mode not in channel_map:
            raise ValueError(f"Unsupported magnetic bridge mode for physics regularizer: {bridge_mode}")

        observation_operator = ChannelProjectionOperator(channel_index=channel_map[bridge_mode])
        dcf_operator = DCFMagneticOperator(
            column_density_index=channel_map["column_density"],
            angle_dispersion_index=channel_map["angle_dispersion"],
            velocity_dispersion_index=channel_map["los_velocity_dispersion"],
            observation_stats=observation_stats,
            target_stats=target_stats,
            q_factor=getattr(opt, "physics_mag_dcf_q", 1.0),
        )
        return MagneticPhysicsRegularizer(
            w_smooth=getattr(opt, "physics_mag_smooth_weight", 0.0),
            w_in_range=getattr(opt, "physics_mag_range_weight", 0.0),
            w_obs_consistency=getattr(opt, "physics_mag_observation_weight", 0.0),
            w_grad_consistency=getattr(opt, "physics_mag_gradient_weight", 0.0),
            w_dcf_consistency=getattr(opt, "physics_mag_dcf_weight", 0.0),
            observation_operator=observation_operator,
            dcf_operator=dcf_operator,
            target_stats=target_stats,
            observation_stats=observation_stats,
            obs_noise_scale=getattr(opt, "obs_noise_scale", 1.0),
            time_weight_power=getattr(opt, "physics_time_weight_power", 1.0),
            total_steps=getattr(opt, "interval", 1000),
            edge_scale=getattr(opt, "physics_mag_edge_scale", 8.0),
        )

    raise ValueError(f"Unsupported task for physics regularizer: {task}")
