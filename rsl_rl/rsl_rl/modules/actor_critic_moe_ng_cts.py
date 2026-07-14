from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any

from tensordict import TensorDict
from rsl_rl.utils import resolve_nn_activation

from .actor_critic_cts import L2Norm, SimNorm, ActorCriticCTS

class StudentMoEEncoder(nn.Module):
    def __init__(
        self,
        expert_dim:int,
        gating_dim:int,
        hidden_dims:tuple[int, ...]| list[int] = (512, 256),
        expert_num: int = 8,
        expert_hidden_dim: int = 256,
        latent_dim: int = 32,
        activation: str = "elu",
        norm_type: str = "l2norm",
        simnorm_group_dim: int = 8,
        ) -> None:
        super().__init__()

        if expert_dim < 1:
           raise ValueError("expert_dim must be greater than zero.")
        if gating_dim < 1:
           raise ValueError("gating_dim must be greater than zero.")
        if len(hidden_dims) == 0:
           raise ValueError("hidden_dims must contain at least one layer.")
        if expert_num < 1:
           raise ValueError("expert_num must be greater than zero.")
        if norm_type not in ("l2norm", "simnorm"):
           raise ValueError(f"Unsupported latent normalization: {norm_type}.")

        self.expert_dim = expert_dim
        self.gating_dim = gating_dim
        self.expert_num = expert_num
        self.latent_dim = latent_dim

        expert_layers: list[nn.Module] = []
        last_dim = expert_dim

        # Shared feature extractor for histories without command inputs.
        for hidden_dim in hidden_dims:
           expert_layers.append(nn.Linear(last_dim, hidden_dim))
           expert_layers.append(resolve_nn_activation(activation))
           last_dim = hidden_dim

        self.experts_backbone = nn.Sequential(*expert_layers)

        # Produce one separate hidden vector for every expert.
        self.experts_hidden = nn.Sequential(
           nn.Linear(last_dim, expert_num * expert_hidden_dim),
           resolve_nn_activation(activation)
        )

        # Equivalent to expert_num independent linear output heads.
        self.experts_out = nn.Conv1d(
           in_channels=expert_num * expert_hidden_dim,
           out_channels=expert_num * latent_dim,
           kernel_size=1,
           groups=expert_num,
        )

        # The gate sees the complete history, including commands.
        gating_layers: list[nn.Module] = []
        last_dim = gating_dim

        for hidden_dim in hidden_dims:
            gating_layers.append(nn.Linear(last_dim, hidden_dim))
            gating_layers.append(resolve_nn_activation(activation))
            last_dim = hidden_dim

        gating_layers.append(nn.Linear(last_dim, expert_num))
        gating_layers.append(nn.Softmax(dim=-1))
        self.gating_network = nn.Sequential(*gating_layers)

        self.norm_layer = (
           L2Norm() if norm_type == "l2norm" else SimNorm(simnorm_group_dim)
        )

    def forward(self, full_history: torch.Tensor, no_goal_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # full_history:[B, history_length × num_actor_obs] ->  [B, expert_num]
        weights = self.gating_network(full_history)

        shared_features = self.experts_backbone(no_goal_history)
        expert_hidden = self.experts_hidden(shared_features)
        # [batch, expert_num * expert_hidden_dim] -> [batch, expert_num * expert_hidden_dim, 1]
        # -> [batch, expert_dim, 1] -> [batch, expert_dim]
        expert_latent_flat = self.experts_out(expert_hidden.unsqueeze(-1)).squeeze(-1)
        # [batch, expert_dim] -> [batch, expert_num, latent_dim]
        expert_latents = expert_latent_flat.reshape(*full_history.shape[:-1], self.expert_num, self.latent_dim,)
        # [batch, expert_num, latent_dim] -> [batch, latent_dim]
        latent = torch.sum(weights.unsqueeze(-1)* expert_latents, dim=-2,)

        latent = self.norm_layer(latent)

        return latent, weights

class ActorCriticMoENGCTS(ActorCriticCTS):
    """CTS actor-critic using a no-goal Student MoE encoder."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        obs_no_goal_mask: tuple[bool, ...] | list[bool],
        history_length: int = 5,
        student_encoder_hidden_dims: tuple[int, ...] | list[int] = (512, 256,),
        student_expert_num: int = 8,
        student_expert_hidden_dim: int = 256,
        activation: str = "elu",
        latent_dim: int = 32,
        norm_type: str = "l2norm",
        simnorm_group_dim: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            num_actions=num_actions,
            history_length=history_length,
            student_encoder_hidden_dims=student_encoder_hidden_dims,
            activation=activation,
            latent_dim=latent_dim,
            norm_type=norm_type,
            simnorm_group_dim=simnorm_group_dim,
            **kwargs,
        )

        mask = torch.as_tensor(obs_no_goal_mask, dtype=torch.bool)

        if mask.ndim != 1:
            raise ValueError(
                "obs_no_goal_mask must be a one-dimensional mask."
            )

        if mask.numel() != self.num_actor_obs:
            raise ValueError(
            "obs_no_goal_mask contains "
            f"{mask.numel()} elements, but policy observation "
            f"contains {self.num_actor_obs} elements."
        )

        if not mask.any():
            raise ValueError(
                "obs_no_goal_mask must keep at least one observation."
            )

        self.register_buffer(
            "obs_no_goal_mask",
            mask,
            persistent=False,
        )

        self.num_expert_obs = int(mask.sum().item())

        expert_input_dim = (
            history_length * self.num_expert_obs
        )

        gating_input_dim = (
            history_length * self.num_actor_obs
        )

        # Remove the standard CTS Student Encoder created by the base class.
        del self.student_encoder

        self.student_moe_encoder = StudentMoEEncoder(
            expert_dim=expert_input_dim,
            gating_dim=gating_input_dim,
            hidden_dims=student_encoder_hidden_dims,
            expert_num=student_expert_num,
            expert_hidden_dim=student_expert_hidden_dim,
            latent_dim=latent_dim,
            activation=activation,
            norm_type=norm_type,
            simnorm_group_dim=simnorm_group_dim,
        )

        print(f"Student MoE Encoder: {self.student_moe_encoder}")

    def _encode_prepared_history(self, full_history: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
        history_frames = full_history.reshape(
            *full_history.shape[:-1],
            self.history_length,
            self.num_actor_obs,
        )

        no_goal_history = history_frames[..., self.obs_no_goal_mask].flatten(start_dim=-2) # type:ignore

        return self.student_moe_encoder(full_history, no_goal_history)

    def get_student_latent_and_weights(self, history: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
        full_history = self._prepare_history(history)
        return self._encode_prepared_history(full_history)

    def get_student_latent(self, history: torch.Tensor,)-> torch.Tensor:
        latent, _ = self.get_student_latent_and_weights(history)
        return latent

    def act_inference(self, obs:TensorDict,) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)

        expected_shape = (actor_obs.shape[0], self.history_length, self.num_actor_obs,)

        if tuple(self._inference_history.shape) != expected_shape:
            self._inference_history = torch.zeros(
            expected_shape,
            device=actor_obs.device,
            dtype=actor_obs.dtype,
        )

        self._inference_history = torch.cat(
            (
                self._inference_history[:, 1:],
                actor_obs.unsqueeze(1),
            ),
            dim=1,
        )

        full_history = self._inference_history.flatten(start_dim=1)
        latent, _ = self._encode_prepared_history(full_history)
        actor_input = torch.cat((actor_obs, latent), dim=-1)

        return self.actor(actor_input)
