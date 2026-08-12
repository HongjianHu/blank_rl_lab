from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from rsl_rl.networks import MLP

def make_activation(name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }

    if name not in activations:
        raise ValueError(
            f"Unsupported activation '{name}'. "
            f"Available activations: {list(activations)}"
        )

    return activations[name]()

class TerrainScanEncoder(nn.Module):
    """将132维高度扫描压缩为32维地形特征。"""
    def __init__(self, input_dim: int, hidden_dims: list[int], activation: str, output_tanh: bool = True) -> None:
        super().__init__()

        if len(hidden_dims) == 0:
            raise ValueError("Terrain encoder hidden_dims cannot be empty.")

        layers: list[nn.Module] = []
        previous_dim = input_dim

        for index, output_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(previous_dim, output_dim))

            is_last_layer = index == len(hidden_dims) - 1

            if is_last_layer and output_tanh:
               layers.append(nn.Tanh())
            else:
               layers.append(make_activation(activation))

            previous_dim = output_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, terrain_scan: torch.Tensor) -> torch.Tensor:
        return self.network(terrain_scan)

class PrivilegedLatentEncoder(nn.Module):
    """将29维动力学特权参数压缩为20维latent。"""
    def __init__(self, input_dim: int, hidden_dims: list[int], activation: str) -> None:
        super().__init__()

        if len(hidden_dims) == 0:
           raise ValueError("Privileged encoder hidden_dims cannot be empty.")

        layers: list[nn.Module] = []
        previous_dim = input_dim

        for output_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, output_dim))
            layers.append(make_activation(activation))
            previous_dim = output_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, privileged: torch.Tensor) -> torch.Tensor:
        return self.network(privileged)

class ExtremeParkourStateHistoryEncoder(nn.Module):
    """官方10帧本体感觉历史编码器。
    input:[batch, 10, 53]
    output:[batch, 20]
    """
    def __init__(self, proprio_dim: int, history_length: int, output_dim: int, activation: str) -> None:
        super().__init__()
        frame_feature_dim = 30

        self.frame_encoder = nn.Sequential(nn.Linear(proprio_dim, frame_feature_dim), make_activation(activation))

        # 输入卷积前的形状为[batch, 30, 10] ->  [batch, 20, 4] -> [batch, 10, 3] - > [batch, 30]
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(in_channels=30, out_channels=20, kernel_size=4, stride=2),
            make_activation(activation),
            nn.Conv1d(in_channels=20, out_channels=10, kernel_size=2, stride=1),
            make_activation(activation),
            nn.Flatten())

        self.output_layer = nn.Sequential(nn.Linear(30, output_dim), make_activation(activation),)

        self.proprio_dim = proprio_dim
        self.history_length = history_length
        self.output_dim = output_dim

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        batch_size = history.shape[0]

        # [B, 10, 53] -> [B*10, 53]
        history = history.reshape(batch_size * self.history_length, self.proprio_dim)

        # [B*10, 53] -> [B*10, 30]
        frame_features = self.frame_encoder(history)

        frame_features = frame_features.reshape(batch_size, self.history_length, 30)

        frame_features = frame_features.permute(0, 2, 1).contiguous()

        temporal_features = self.temporal_encoder(frame_features)

        return self.output_layer(temporal_features)

class ExtremeParkourActor(nn.Module):
    def __init__(
        self,
        proprio_dim: int,
        terrain_scan_dim: int,
        priv_explicit_dim: int,
        priv_latent_dim: int,
        history_length: int,
        scan_encoder_dims: list[int],
        priv_encoder_dims: list[int],
        history_latent_dim: int,
        actor_hidden_dims: list[int],
        num_actions: int,
        activation: str,
        scan_encoder_output_tanh: bool,
    ) -> None:
        super().__init__()
        self.proprio_dim = proprio_dim
        self.history_length = history_length

        self.scan_encoder = TerrainScanEncoder(
            input_dim=terrain_scan_dim,
            hidden_dims=scan_encoder_dims,
            activation=activation,
            output_tanh=scan_encoder_output_tanh,
        )

        self.priv_encoder = PrivilegedLatentEncoder(
            input_dim=priv_latent_dim,
            hidden_dims=priv_encoder_dims,
            activation=activation,
        )
        # [B, 10, 53] -> [B, 20]
        self.history_encoder = ExtremeParkourStateHistoryEncoder(
            proprio_dim=proprio_dim,
            history_length=history_length,
            output_dim=history_latent_dim,
            activation=activation,
        )

        actor_input_dim =  proprio_dim + self.scan_encoder.output_dim + priv_explicit_dim +  self.priv_encoder.output_dim

        self.backbone = MLP(actor_input_dim, num_actions, actor_hidden_dims, activation)

        self.input_dim = actor_input_dim
        self.num_actions = num_actions

    def encode_scan(self, terrain_scan: torch.Tensor) -> torch.Tensor:
        return self.scan_encoder(terrain_scan)

    def encode_privileged(self, priv_latent: torch.Tensor) -> torch.Tensor:
        return self.priv_encoder(priv_latent)

    def encode_history(self, proprio_history: torch.Tensor) -> torch.Tensor:
        batch_size = proprio_history.shape[0]

        history = proprio_history.reshape(batch_size, self.history_length, self.proprio_dim)

        return self.history_encoder(history)

    def forward(
        self,
        proprio: torch.Tensor,
        terrain_scan: torch.Tensor,
        priv_explicit: torch.Tensor,
        priv_latent: torch.Tensor,
        proprio_history: torch.Tensor,
        use_history: bool = False,
        scan_latent_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """计算高斯动作分布的均值。
        use_history=False:
            使用29维特权参数经过编码得到的20维latent。

        use_history=True:
            使用10帧本体历史得到的20维latent。
        """
        if scan_latent_override is None:
           scan_latent = self.encode_scan(terrain_scan)
        else:
           scan_latent = scan_latent_override

        if use_history:
           adaptation_latent = self.encode_history(proprio_history)
        else:
           adaptation_latent = self.encode_privileged(priv_latent)

        actor_input = torch.cat((proprio, scan_latent, priv_explicit, adaptation_latent), dim=-1)

        return self.backbone(actor_input)

class ExtremeParkourEstimator(nn.Module):
    """从53维本体感觉估计9维显式特权状态。"""

    def __init__(
        self,
        input_dim: int = 53,
        hidden_dims: list[int] = [128, 64],
        output_dim: int = 9,
        activation: str = "elu",
    ) -> None:
        super().__init__()

        self.network = MLP(input_dim, output_dim, hidden_dims, activation)

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.network(proprio)

class ExtremeParkourActorCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        # ======================= #
        proprio_dim: int = 53,
        terrain_scan_dim: int = 132,
        priv_explicit_dim: int = 9,
        priv_latent_dim: int = 29,
        history_length: int = 10,
        proprio_history_dim: int = 530,
        scan_encoder_dims: list[int] = [128, 64, 32],
        priv_encoder_dims: list[int] = [64, 20],
        history_latent_dim: int = 20,
        actor_input_dim: int = 114,
        critic_input_dim: int = 753,
        scan_encoder_output_tanh: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if kwargs:
            raise ValueError(
                "Unexpected ExtremeParkourActorCritic arguments: "
                f"{list(kwargs)}"
            )

        if actor_obs_normalization or critic_obs_normalization:
            raise ValueError(
                "Extreme Parkour currently uses observation scaling in the "
                "environment and does not enable empirical normalization."
            )

        if state_dependent_std:
            raise ValueError(
                "Extreme Parkour uses one learned action standard-deviation "
                "vector, not state-dependent standard deviation."
            )

        self.obs_groups = obs_groups
        self.num_actions = num_actions
        self.noise_std_type = noise_std_type

        self._resolve_observation_groups()
        self._validate_observation_contract(
            obs=obs,
            proprio_dim=proprio_dim,
            terrain_scan_dim=terrain_scan_dim,
            priv_explicit_dim=priv_explicit_dim,
            priv_latent_dim=priv_latent_dim,
            proprio_history_dim=proprio_history_dim,
            critic_input_dim=critic_input_dim,
        )

        self.actor = ExtremeParkourActor(
            proprio_dim=proprio_dim,
            terrain_scan_dim=terrain_scan_dim,
            priv_explicit_dim=priv_explicit_dim,
            priv_latent_dim=priv_latent_dim,
            history_length=history_length,
            scan_encoder_dims=scan_encoder_dims,
            priv_encoder_dims=priv_encoder_dims,
            history_latent_dim=history_latent_dim,
            actor_hidden_dims=actor_hidden_dims,
            num_actions=num_actions,
            activation=activation,
            scan_encoder_output_tanh=scan_encoder_output_tanh,
        )


        self.critic = MLP(critic_input_dim, 1, critic_hidden_dims, activation)

        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(
                "noise_std_type must be either 'scalar' or 'log', "
                f"but received '{noise_std_type}'."
            )
        self.distribution: Normal | None = None

        Normal.set_default_validate_args(False)

    def _resolve_observation_groups(self) -> None:
        """把算法集合解析成实际TensorDict键名。"""

        policy_groups = self.obs_groups["policy"]

        (self.proprio_key, self.terrain_scan_key, self.priv_explicit_key, self.priv_latent_key) = policy_groups

        adaptation_groups = self.obs_groups["adaptation"]
        estimator_groups = self.obs_groups["estimator"]
        estimator_target_groups = self.obs_groups["estimator_target"]

        if len(adaptation_groups) != 1:
            raise ValueError("'adaptation' must contain exactly one group.")

        if len(estimator_groups) != 1:
            raise ValueError("'estimator' must contain exactly one group.")

        if len(estimator_target_groups) != 1:
            raise ValueError(
                "'estimator_target' must contain exactly one group."
            )

        self.history_key = adaptation_groups[0]
        self.estimator_input_key = estimator_groups[0]
        self.estimator_target_key = estimator_target_groups[0]
        self.critic_keys = self.obs_groups["critic"]

    @staticmethod
    def _validate_tensor(
        obs: TensorDict,
        key: str,
        expected_dim: int,
    ) -> None:
        if key not in obs:
            raise KeyError(
                f"Observation group '{key}' is missing. "
                f"Available groups: {list(obs.keys())}"
            )

        if obs[key].ndim != 2:
            raise ValueError(
                f"Observation '{key}' must have shape [num_envs, dim], "
                f"but received {tuple(obs[key].shape)}."
            )

        if obs[key].shape[-1] != expected_dim:
            raise ValueError(
                f"Observation '{key}' must be {expected_dim}-D, "
                f"but received {obs[key].shape[-1]}-D."
            )

    def _validate_observation_contract(
        self,
        obs: TensorDict,
        proprio_dim: int,
        terrain_scan_dim: int,
        priv_explicit_dim: int,
        priv_latent_dim: int,
        proprio_history_dim: int,
        critic_input_dim: int,
    ) -> None:
        self._validate_tensor(obs, self.proprio_key, proprio_dim)

        self._validate_tensor(obs, self.terrain_scan_key, terrain_scan_dim)

        self._validate_tensor(obs, self.priv_explicit_key, priv_explicit_dim)

        self._validate_tensor(obs, self.priv_latent_key, priv_latent_dim)

        self._validate_tensor(obs, self.history_key, proprio_history_dim)

        calculated_critic_dim = sum(obs[key].shape[-1] for key in self.critic_keys)

        if calculated_critic_dim != critic_input_dim:
            raise ValueError(
                f"Critic observation must be {critic_input_dim}-D, "
                f"but configured critic groups produce "
                f"{calculated_critic_dim}-D."
            )

        if self.estimator_input_key != self.proprio_key:
            raise ValueError(
                "The estimator input must currently reference proprio."
            )

        if self.estimator_target_key != self.priv_explicit_key:
            raise ValueError(
                "The estimator target must currently reference "
                "priv_explicit."
            )

    def _actor_mean(
        self,
        obs: TensorDict,
        use_history: bool,
        explicit_override: torch.Tensor | None,
        scan_latent_override: torch.Tensor | None,
    ) -> torch.Tensor:
        if explicit_override is None:
            explicit_state = obs[self.priv_explicit_key]
        else:
            explicit_state = explicit_override

        return self.actor(
            proprio=obs[self.proprio_key],
            terrain_scan=obs[self.terrain_scan_key],
            priv_explicit=explicit_state,
            priv_latent=obs[self.priv_latent_key],
            proprio_history=obs[self.history_key],
            use_history=use_history,
            scan_latent_override=scan_latent_override,
        )

    def _get_std(self) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            with torch.no_grad():
                clean_std = torch.nan_to_num(
                    self.std,
                    nan=1.0,
                    posinf=10.0,
                    neginf=1.0,
                )
                self.std.copy_(clean_std.clamp(min=1.0e-6, max=10.0))

            return self.std

        return torch.exp(self.log_std.clamp(min=-20.0, max=2.0))

    def _update_distribution(
        self,
        obs: TensorDict,
        use_history: bool,
        explicit_override: torch.Tensor | None,
        scan_latent_override: torch.Tensor | None,
    ) -> None:
        mean = self._actor_mean(
            obs=obs,
            use_history=use_history,
            explicit_override=explicit_override,
            scan_latent_override=scan_latent_override,
        )

        std = self._get_std().expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(
        self,
        obs: TensorDict,
        use_history: bool = False,
        explicit_override: torch.Tensor | None = None,
        scan_latent_override: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """训练时从高斯策略中采样动作。"""

        self._update_distribution(
            obs=obs,
            use_history=use_history,
            explicit_override=explicit_override,
            scan_latent_override=scan_latent_override,
        )

        return self.distribution.sample()  # type: ignore[union-attr]

    def act_inference(
        self,
        obs: TensorDict,
        use_history: bool = False,
        explicit_override: torch.Tensor | None = None,
        scan_latent_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """评测时直接返回动作分布均值，不再加入随机噪声。"""

        return self._actor_mean(
            obs=obs,
            use_history=use_history,
            explicit_override=explicit_override,
            scan_latent_override=scan_latent_override,
        )

    def evaluate(
        self,
        obs: TensorDict,
        **kwargs: Any,
    ) -> torch.Tensor:
        """计算Critic状态价值V(s)。"""

        critic_obs = torch.cat(
            [obs[key] for key in self.critic_keys],
            dim=-1,
        )

        return self.critic(critic_obs)

    def infer_scan_latent(
        self,
        obs: TensorDict,
    ) -> torch.Tensor:
        return self.actor.encode_scan(obs[self.terrain_scan_key])

    def infer_priv_latent(
        self,
        obs: TensorDict,
    ) -> torch.Tensor:
        return self.actor.encode_privileged(obs[self.priv_latent_key])

    def infer_history_latent(
        self,
        obs: TensorDict,
    ) -> torch.Tensor:
        return self.actor.encode_history(obs[self.history_key])

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError(
                "Action distribution has not been initialized."
            )

        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError(
                "Action distribution has not been initialized."
            )

        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError(
                "Action distribution has not been initialized."
            )

        return self.distribution.entropy().sum(dim=-1)

    def get_actions_log_prob(
        self,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError(
                "Action distribution has not been initialized."
            )

        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        return None

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """教师网络本身没有循环隐状态,无需按环境reset。"""
        return None

    def forward(self) -> NoReturn:
        raise NotImplementedError("Use act(), act_inference(), or evaluate().")

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True
