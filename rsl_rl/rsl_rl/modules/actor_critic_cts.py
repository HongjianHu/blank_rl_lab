from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization

class L2Norm(nn.Module):
    """Normalize each latent vector to unit L2 length."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2.0, dim=-1)

class SimNorm(nn.Module):
    """Apply softmax independently to groups inside the latent vector."""

    def __init__(self, group_dim: int = 8) -> None:
        super().__init__()
        self.group_dim = group_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % self.group_dim != 0:
            raise ValueError(
                f"Latent dimension {x.shape[-1]} must be divisible "
                f"by SimNorm group dimension {self.group_dim}."
            )

        original_shape = x.shape
        x = x.reshape(*original_shape[:-1], -1, self.group_dim)
        x = F.softmax(x, dim=-1)
        return x.reshape(original_shape)

class ActorCriticCTS(nn.Module):
    """Concurrent teacher-student actor-critic network."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        history_length: int = 5,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int, ...] | list[int] = [512, 256, 128],
        critic_hidden_dims: tuple[int, ...] | list[int] = [512, 256, 128],
        teacher_encoder_hidden_dims: tuple[int, ...] | list[int] = [512, 256],
        student_encoder_hidden_dims: tuple[int, ...] | list[int] = [512, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        latent_dim: int = 32,
        norm_type: str = "l2norm",
        simnorm_group_dim: int = 8,
        **kwargs: dict[str, Any],
    )-> None:
        if kwargs:
            print(
                "ActorCriticCTS.__init__ got unexpected arguments, "
                f"which will be ignored: {list(kwargs)}"
            )
        super().__init__()

        if norm_type not in ("l2norm", "simnorm"):
            raise ValueError(f"Unsupported latent normalization: {norm_type}.")

        if history_length < 1:
            raise ValueError("history_length must be greater than zero.")

        self.obs_groups = obs_groups
        self.history_length = history_length
        self.num_actions = num_actions

        self.num_actor_obs = self._resolve_obs_dim(
            obs, obs_groups["policy"]
        )

        self.num_critic_obs = self._resolve_obs_dim(
            obs, obs_groups["critic"]
        )

        student_input_dim = self.num_actor_obs * history_length
        actor_input_dim = self.num_actor_obs + latent_dim
        critic_input_dim = self.num_critic_obs + latent_dim

        def latent_normalizer():
            if norm_type == "l2norm":
                return L2Norm()
            return SimNorm(simnorm_group_dim)

        self.teacher_encoder = nn.Sequential(
            MLP(
                self.num_critic_obs,
                latent_dim,
                teacher_encoder_hidden_dims,
                activation,
            ),
            latent_normalizer(),
        )

        self.student_encoder = nn.Sequential(
            MLP(
                student_input_dim,
                latent_dim,
                student_encoder_hidden_dims,
                activation,
            ),
            latent_normalizer(),
        )

        self.actor = MLP(
            actor_input_dim,
            num_actions,
            actor_hidden_dims,
            activation,
        )

        self.critic = MLP(
            critic_input_dim,
            1,
            critic_hidden_dims,
            activation,
        )

        self.actor_obs_normalization = actor_obs_normalization
        self.critic_obs_normalization = critic_obs_normalization

        self.actor_obs_normalizer = (
            EmpiricalNormalization(self.num_actor_obs)
            if actor_obs_normalization
            else nn.Identity()
        )

        self.critic_obs_normalizer = (
            EmpiricalNormalization(self.num_critic_obs)
            if critic_obs_normalization
            else nn.Identity()
        )

        print(f"Teacher Encoder: {self.teacher_encoder}")
        print(f"Student Encoder: {self.student_encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.noise_std_type = noise_std_type

        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError(
                f"Unsupported noise_std_type: {noise_std_type}. "
                "Expected 'scalar' or 'log'."
            )

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

        self.register_buffer(
            "_inference_history",
            torch.empty(0),
            persistent=False
        )

    @staticmethod
    def _resolve_obs_dim(
        obs: TensorDict,
        group_names: list[str],
        ) -> int:
        obs_dim = 0
        for group_name in group_names:
            if len(obs[group_name].shape) != 2:
                raise ValueError(
                    "ActorCriticCTS only supports vector observations, "
                    f"but '{group_name}' has shape {obs[group_name].shape}."
                )
            obs_dim += obs[group_name].shape[-1]
        return obs_dim

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat(
            [obs[name] for name in self.obs_groups["policy"]],
            dim=-1,
        )

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat(
            [obs[name] for name in self.obs_groups["critic"]],
            dim=-1,
        )

    def _prepare_history(self, history: torch.Tensor) -> torch.Tensor:
        expected_flat_dim = self.history_length * self.num_actor_obs

        if tuple(history.shape[-2:]) == (
            self.history_length,
            self.num_actor_obs
         ):
            history_frames = history

        elif history.shape[-1] == expected_flat_dim:
            history_frames = history.reshape(
                *history.shape[:-1],
                self.history_length,
                self.num_actor_obs,
            )
        else:
            raise ValueError(
                f"Invalid history shape {history.shape}. Expected "
                f"(..., {self.history_length}, {self.num_actor_obs}) "
                f"or (..., {expected_flat_dim})."
        )

        flat_frames = history_frames.reshape(-1, self.num_actor_obs)
        flat_frames = self.actor_obs_normalizer(flat_frames)

        return flat_frames.reshape(
            *history_frames.shape[:-2],
            expected_flat_dim,
        )

    def get_teacher_latent(self, obs: TensorDict) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        return self.teacher_encoder(critic_obs)

    def get_student_latent(self, history: torch.Tensor) -> torch.Tensor:
        history = self._prepare_history(history)
        return self.student_encoder(history)

    def _update_distribution(self, actor_input: torch.Tensor) -> None:
        action_mean = self.actor(actor_input)

        if self.noise_std_type == "scalar":
            with torch.no_grad():
                action_std = self.std.clamp(
                    min=1.0e-6,
                    max=10.0,
                ).expand_as(action_mean)
        else:
            log_std = self.log_std.clamp(min=-20.0, max=2.0)
            action_std = torch.exp(log_std).expand_as(action_mean)

        self.distribution = Normal(action_mean, action_std)

    def act(self, obs:TensorDict, history: torch.Tensor, is_teacher: bool) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)

        if is_teacher:
            latent = self.get_teacher_latent(obs)
        else:
            # Student Encoder不通过PPO policy的loss 更新，
            # 只通过后面CTS algorithm的latent imitation loss更新
            latent = self.get_student_latent(history).detach()

        actor_input = torch.cat([actor_obs, latent], dim=-1)
        self._update_distribution(actor_input)

        return self.distribution.sample() # type:ignore

    def evaluate(self, obs:TensorDict, history: torch.Tensor, is_teacher: bool,) -> torch.Tensor:
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)

        if is_teacher:
           latent = self.teacher_encoder(critic_obs)
        else:
           latent = self.get_student_latent(history)

        critic_input = torch.cat( (critic_obs, latent.detach()), dim=-1,)

        return self.critic(critic_input)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean  # type:ignore

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev # type:ignore

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1) # type:ignore

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)  # type: ignore

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)

        expected_shape = (
            actor_obs.shape[0],
            self.history_length,
            self.num_actor_obs,
        )

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

        history = self._inference_history.flatten(start_dim=1)
        latent = self.student_encoder(history)
        actor_input = torch.cat((actor_obs, latent), dim=-1)

        return self.actor(actor_input)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        if self._inference_history.numel() == 0:
            return
        if dones is None:
           self._inference_history.zero_()
        else:
           done_mask = dones.reshape(-1).bool()
           self._inference_history[done_mask] = 0.0

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs)) # type:ignore

        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs)) # type:ignore

    def load_state_dict(
        self,
        state_dict: dict,
        strict: bool = True,
    ) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
