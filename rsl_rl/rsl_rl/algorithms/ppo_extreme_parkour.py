from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import (
    ExtremeParkourActorCritic,
    ExtremeParkourEstimator,
)
from rsl_rl.storage import RolloutStorage

class ExtremeParkourPPO(PPO):
    """Extreme Parkour教师阶段使用的PPO。

    在普通PPO基础上增加:
    1. 显式状态估计器训练。
    2. 特权latent正则。
    3. 历史适应模块训练。
    4. 特权/历史双路径rollout。
    """

    policy: ExtremeParkourActorCritic

    def __init__(
        self,
        policy: ExtremeParkourActorCritic,
        storage: RolloutStorage,
        estimator_input_dim: int = 53,
        estimator_hidden_dims: list[int] = [128, 64],
        estimator_output_dim: int = 9,
        estimator_learning_rate: float = 1.0e-4,
        train_with_estimated_states: bool = True,
        history_update_interval: int = 20,
        priv_reg_coef_schedule: list[float] = [
            0.0,
            0.1,
            2000.0,
            3000.0,
        ],
        device: str = "cpu",
        **ppo_kwargs,
    ) -> None:
        if ppo_kwargs.get("rnd_cfg") is not None:
            raise ValueError("ExtremeParkourPPO does not use RND.")

        if ppo_kwargs.get("symmetry_cfg") is not None:
            raise ValueError("ExtremeParkourPPO does not currently use symmetry.")

        if ppo_kwargs.get("multi_gpu_cfg") is not None:
            raise NotImplementedError("The current ExtremeParkourPPO implementation supports single-GPU training only.")

        super().__init__(
            policy=policy, # type:ignore
            storage=storage,
            device=device,
            **ppo_kwargs,
        )

        self.policy = policy
        self.device = device

        self.estimator = ExtremeParkourEstimator(
            input_dim=estimator_input_dim,
            hidden_dims=estimator_hidden_dims,
            output_dim=estimator_output_dim,
            activation="elu",
        ).to(device)

        self.estimator_optimizer = torch.optim.Adam(
            self.estimator.parameters(),
            lr=estimator_learning_rate,
        )

        self.history_optimizer = torch.optim.Adam(
            self.policy.actor.history_encoder.parameters(),
            lr=self.learning_rate,
        )

        self.train_with_estimated_states = train_with_estimated_states

        if history_update_interval <= 0:
            raise ValueError("history_update_interval must be positive.")
        self.history_update_interval = history_update_interval

        if len(priv_reg_coef_schedule) != 4:
            raise ValueError(
                "priv_reg_coef_schedule must contain [initial, final, start_iteration, duration]."
            )

        if priv_reg_coef_schedule[3] <= 0:
            raise ValueError(
                "The regularization schedule duration must be positive."
            )

        self.priv_reg_coef_schedule = priv_reg_coef_schedule

        self.update_counter = 0

        self._validate_estimator_contract(
            storage=storage,
            input_dim=estimator_input_dim,
            output_dim=estimator_output_dim,
        )

    def _validate_estimator_contract(
        self,
        storage: RolloutStorage,
        input_dim: int,
        output_dim: int,
    ) -> None:
        input_key = self.policy.estimator_input_key
        target_key = self.policy.estimator_target_key

        actual_input_dim =  storage.observations[input_key].shape[-1]
        actual_target_dim =  storage.observations[target_key].shape[-1]

        if actual_input_dim != input_dim:
            raise ValueError(
                f"Estimator input must be {input_dim}-D, but storage contains {actual_input_dim}-D."
            )

        if actual_target_dim != output_dim:
            raise ValueError(
                f"Estimator target must be {output_dim}-D, but storage contains {actual_target_dim}-D."
            )

    def _history_update_due(self) -> bool:
        next_update_number = self.update_counter + 1

        return next_update_number % self.history_update_interval == 0

    def _get_priv_reg_coef(self) -> float:
        initial, final, start, duration = self.priv_reg_coef_schedule

        progress = (self.update_counter - start) / duration

        progress = max(0.0, min(1.0, progress))

        return initial + progress * (final - initial)

    def act(self, obs: TensorDict) -> torch.Tensor:
        use_history = self._history_update_due()

        if self.train_with_estimated_states:
            with torch.inference_mode():
              explicit_estimate = self.estimator(obs[self.policy.estimator_input_key])
        else:
            explicit_estimate = None

        self.transition.actions = self.policy.act(obs, use_history=use_history, explicit_override=explicit_estimate).detach() # type:ignore
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach() # type:ignore
        self.transition.action_mean =  self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()

        # 注意：保存的是动作执行前的真实观测。
        self.transition.observations = obs

        return self.transition.actions # type:ignore

    def _update_learning_rate(
        self,
        mu_batch: torch.Tensor,
        sigma_batch: torch.Tensor,
        old_mu_batch: torch.Tensor,
        old_sigma_batch: torch.Tensor,
    ) -> float:
        if (self.desired_kl is None or self.schedule != "adaptive"):
            return 0.0

        with torch.inference_mode():
            sigma = sigma_batch.clamp_min(1.0e-6)
            old_sigma = old_sigma_batch.clamp_min(1.0e-6)

            kl = torch.sum(
                torch.log(sigma / old_sigma)
                + (
                    old_sigma.square()
                    + (old_mu_batch - mu_batch).square()
                )
                / (2.0 * sigma.square())
                - 0.5,
                dim=-1,
            )

            kl_mean = kl.mean()

        if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5,)

        elif (
            kl_mean < self.desired_kl / 2.0
            and kl_mean > 0.0
        ):
            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5,
            )

        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = self.learning_rate

        return kl_mean.item()

    def _update_estimator(self, obs_batch: TensorDict) -> torch.Tensor:
        '''通过本体感知53d预测线速度9d'''
        proprio = obs_batch[self.policy.estimator_input_key]

        target = obs_batch[self.policy.estimator_target_key]

        prediction = self.estimator(proprio)

        estimator_loss = F.mse_loss(prediction, target)

        self.estimator_optimizer.zero_grad(set_to_none=True)

        estimator_loss.backward()

        nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)

        self.estimator_optimizer.step()

        return estimator_loss.detach()

    def _update_history_encoder(self) -> float:
        """通过历史本体感知预测特权信息"""
        mean_history_loss = 0.0
        num_updates = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            _actions_batch,
            _target_values_batch,
            _advantages_batch,
            _returns_batch,
            _old_log_prob_batch,
            _old_mu_batch,
            _old_sigma_batch,
            _hidden_states_batch,
            _masks_batch,
        ) in generator:
            with torch.inference_mode():
                priv_latent = self.policy.infer_priv_latent(obs_batch)

            history_latent = self.policy.infer_history_latent(obs_batch)

            history_loss = torch.linalg.vector_norm(history_latent - priv_latent.detach(), ord=2, dim=-1).mean()

            self.history_optimizer.zero_grad(set_to_none=True)

            history_loss.backward()

            nn.utils.clip_grad_norm_(self.policy.actor.history_encoder.parameters(), self.max_grad_norm)

            self.history_optimizer.step()

            mean_history_loss += history_loss.item()
            num_updates += 1

        return mean_history_loss / max(num_updates, 1)

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_estimator_loss = 0.0
        mean_priv_reg_loss = 0.0
        mean_kl = 0.0
        num_updates = 0

        priv_reg_coef = self._get_priv_reg_coef()
        history_update_due = self._history_update_due()

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
        )

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            del hidden_states_batch, masks_batch

            if self.normalize_advantage_per_mini_batch:
                advantages_batch = (
                    advantages_batch
                    - advantages_batch.mean()
                ) / (
                    advantages_batch.std()
                    + 1.0e-8
                )

            # PPO更新使用真实priv_explicit和priv_latent分支。
            # 主要是为了用真实特权信息训练一个性能尽可能高、训练尽可能稳定的教师策略。
            # 教师学好后，再让其他模块模仿教师需要的信息：
            # Estimator                proprio → 模仿真实priv_explicit
            # History Encoder  proprio_history → 模仿privileged latent
            # 视觉学生                    depth → 模仿terrain scan latent和教师动作
            self.policy.act(obs_batch, use_history=False, explicit_override=None) # 得到新的动作

            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)

            value_batch = self.policy.evaluate(obs_batch)

            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            kl_value = self._update_learning_rate(
                mu_batch=mu_batch,
                sigma_batch=sigma_batch,
                old_mu_batch=old_mu_batch,
                old_sigma_batch=old_sigma_batch,
            )

            old_log_prob = old_actions_log_prob_batch.squeeze(-1)
            advantages = advantages_batch.squeeze(-1)

            probability_ratio = torch.exp(actions_log_prob_batch- old_log_prob)

            surrogate_unclipped = -advantages * probability_ratio

            surrogate_clipped = -advantages * probability_ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)

            surrogate_loss = torch.maximum(surrogate_unclipped, surrogate_clipped).mean()
            # returns_batch：根据实际奖励和 GAE 计算出的价值监督目标
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,self.clip_param)

                value_loss_unclipped = (value_batch - returns_batch).square()

                value_loss_clipped = (value_clipped - returns_batch).square()

                value_loss = torch.maximum(value_loss_unclipped,value_loss_clipped).mean()

            else:
                value_loss = F.mse_loss(value_batch, returns_batch)

            priv_latent = self.policy.infer_priv_latent(obs_batch)

            with torch.inference_mode():
                history_latent = self.policy.infer_history_latent(obs_batch)

            # 为什么让特权 latent 靠近预测 latent
            # 如果只考虑 PPO 奖励，Privileged Encoder 可能学出一些虽然有利于跑酷，但很难从机器人历史观测中推断的特征
            priv_reg_loss = torch.linalg.vector_norm(priv_latent - history_latent.detach(), ord=2, dim=-1).mean()

            estimator_loss = self._update_estimator(obs_batch)
            # PPO任务损失要求priv latent包含对跑酷有用的信息
            # priv_reg正则 要求priv latent能够被历史观测近似
            total_loss = surrogate_loss + self.value_loss_coef * value_loss \
                         - self.entropy_coef * entropy_batch.mean() + priv_reg_coef * priv_reg_loss
            '''
            proprio
            terrain_scan → Terrain Encoder
            priv_latent → Privileged Encoder
            priv_explicit
                   │
                   ▼
            Actor
                   │
                   ▼
            动作分布
                   │
                   ▼
            log_prob
                   │
                   ▼
            surrogate_loss -> updata TerrainScanEncoder、PrivilegedLatentEncoder、Actor backbone、动作分布参数std
            '''
            self.optimizer.zero_grad(set_to_none=True)

            total_loss.backward()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

            self.optimizer.step()

            self.policy._get_std() #type:ignore

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_estimator_loss += estimator_loss.item()
            mean_priv_reg_loss += priv_reg_loss.item()
            mean_kl += kl_value
            num_updates += 1

        mean_history_loss = 0.0

        if history_update_due:
            mean_history_loss =  self._update_history_encoder()
        self.update_counter += 1
        self.storage.clear()

        denominator = max(num_updates, 1)

        return {
            "value": mean_value_loss / denominator,
            "surrogate": (
                mean_surrogate_loss / denominator
            ),
            "entropy": mean_entropy / denominator,
            "estimator": (
                mean_estimator_loss / denominator
            ),
            "priv_reg": (
                mean_priv_reg_loss / denominator
            ),
            "priv_reg_coef": priv_reg_coef,
            "history_latent": mean_history_loss,
            "kl": mean_kl / denominator,
            "learning_rate": self.learning_rate,
            "history_rollout": float(
                history_update_due
            ),
        }
