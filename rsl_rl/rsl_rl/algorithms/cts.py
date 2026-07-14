from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import ActorCriticCTS
from rsl_rl.storage import RolloutStorageCTS
from rsl_rl.utils import resolve_optimizer

class CTS:
    """Concurrent teacher-student PPO algorithm."""
    policy: ActorCriticCTS
    storage: RolloutStorageCTS
    def __init__(
        self,
        policy: ActorCriticCTS,
        storage: RolloutStorageCTS,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1.0e-3,
        student_encoder_learning_rate: float = 1.0e-3,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float | None = 0.01,
        teacher_env_ratio: float = 0.75,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        if policy.is_recurrent:
            raise ValueError("CTS currently requires a feed-forward policy.")

        if not 0.0 < teacher_env_ratio < 1.0:
            raise ValueError("teacher_env_ratio must be between 0 and 1.")

        self.device = device
        self.policy = policy.to(device)
        self.storage = storage
        self.transition = RolloutStorageCTS.Transition()

        self.teacher_env_ids = storage.teacher_env_ids
        self.student_env_ids = storage.student_env_ids

        expected_teacher_envs = max(
            int(storage.num_envs * teacher_env_ratio),
            1,
        )

        if storage.num_teacher_envs != expected_teacher_envs:
            raise ValueError(
                f"Storage contains {storage.num_teacher_envs} teacher "
                f"environments, but teacher_env_ratio={teacher_env_ratio} "
                f"requires {expected_teacher_envs}."
            )

        noise_parameter = (
            policy.std if policy.noise_std_type == "scalar" else policy.log_std
        )

        self.ppo_parameters = [
            *policy.teacher_encoder.parameters(),
            *policy.actor.parameters(),
            *policy.critic.parameters(),
            noise_parameter,
        ]

        optimizer_class = resolve_optimizer(optimizer)

        self.optimizer = optimizer_class( # type:ignore
            self.ppo_parameters,
            lr = learning_rate,
        )

        self.student_encoder_module = self._get_student_encoder_module()

        self.student_optimizer = optimizer_class( # type:ignore
            self.student_encoder_module.parameters(),
            lr = student_encoder_learning_rate,
        )

        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.student_encoder_learning_rate = student_encoder_learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is None:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1
        else:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]

    def _get_student_encoder_module(self) -> nn.Module:
        """Return the Student module optimized by latent imitation."""
        return self.policy.student_encoder

    def _compute_student_loss(self, student_obs:TensorDict, student_history:torch.Tensor)->tuple[torch.Tensor, dict[str, torch.Tensor]]:
        student_latent = self.policy.get_student_latent(student_history)

        with torch.no_grad():
            teacher_latent = self.policy.get_teacher_latent(student_obs)

        latent_loss = torch.nn.functional.mse_loss(
            student_latent,
            teacher_latent,
        )

        return latent_loss, {"latent": latent_loss}

    def act(self, obs: TensorDict, history: torch.Tensor) -> torch.Tensor:
        num_envs = self.storage.num_envs
        action_shape = tuple(self.storage.actions_shape)

        actions = torch.empty(num_envs, *action_shape, device=self.device)
        values = torch.empty(num_envs, 1, device=self.device)
        log_prob = torch.empty(num_envs, device=self.device)
        action_mean = torch.empty(num_envs,*action_shape,device=self.device,)
        action_std = torch.empty_like(action_mean)

        groups = (
            (self.teacher_env_ids, True),
            (self.student_env_ids, False),
        )

        for env_ids, is_teacher in groups:
            group_obs = obs[env_ids]
            group_history = history[env_ids]

            group_actions = self.policy.act(group_obs, group_history, is_teacher=is_teacher)

            actions[env_ids] = group_actions.detach()
            values[env_ids] = self.policy.evaluate(group_obs, group_history, is_teacher=is_teacher).detach()

            log_prob[env_ids] = self.policy.get_actions_log_prob(group_actions).detach()

            action_mean[env_ids] = self.policy.action_mean.detach()

            action_std[env_ids] = self.policy.action_std.detach()

        self.transition.observations = obs
        self.transition.history = history
        self.transition.actions = actions # type:ignore
        self.transition.values = values
        self.transition.actions_log_prob = log_prob
        self.transition.action_mean = action_mean
        self.transition.action_sigma = action_std

        return actions

    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor],) -> None:
        self.policy.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if "time_outs" in extras:
            time_outs = extras["time_outs"].to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * time_outs.unsqueeze(-1), dim=1 # type:ignore
            )

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def _evaluate_all(self, obs: TensorDict, history: torch.Tensor,) -> torch.Tensor:
        values = torch.empty(self.storage.num_envs, 1, device=self.device)

        for env_ids, is_teacher in ((self.teacher_env_ids, True), (self.student_env_ids, False),):
            values[env_ids] = self.policy.evaluate(obs[env_ids], history[env_ids], is_teacher=is_teacher,)

        return values

    def compute_returns(self, obs: TensorDict, history: torch.Tensor,) -> None:
        with torch.no_grad():
            last_values = self._evaluate_all(obs, history)

        advantage = torch.zeros_like(last_values)

        for step in reversed(range(self.storage.num_transitions_per_env)):
            if step == self.storage.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.storage.values[step + 1]

            next_is_not_terminal = 1.0 - self.storage.dones[step].float()

            delta = (
                self.storage.rewards[step]
                + self.gamma * next_is_not_terminal * next_values
                - self.storage.values[step]
            )
            # lam = 0, advantage 退化为单步 TD 误差
            advantage = (
                delta + self.gamma * self.lam * next_is_not_terminal * advantage
            )

            self.storage.returns[step] = advantage + self.storage.values[step]

        self.storage.advantages = self.storage.returns - self.storage.values

        if not self.normalize_advantage_per_mini_batch:
            advantages = self.storage.advantages
            self.storage.advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)


    def _forward_batch(self, obs_batch: TensorDict, history_batch: torch.Tensor, actions_batch: torch.Tensor, is_teacher_batch: torch.Tensor,) -> tuple[torch.Tensor, ...]:
        teacher_indices = torch.nonzero(is_teacher_batch, as_tuple=False).squeeze(-1)
        student_indices = torch.nonzero(~is_teacher_batch, as_tuple=False).squeeze(-1)

        ordered_indices = torch.cat((teacher_indices, student_indices))

        restore_order = torch.argsort(ordered_indices)

        def forward_group(indices: torch.Tensor, is_teacher: bool,) -> tuple[torch.Tensor, ...]:
            group_obs = obs_batch[indices]
            group_history = history_batch[indices]

            self.policy.act(
                group_obs,
                group_history,
                is_teacher=is_teacher
            )

            return (
                self.policy.get_actions_log_prob(
                    actions_batch[indices]
                ),
                self.policy.evaluate(
                    group_obs,
                    group_history,
                    is_teacher=is_teacher
                ),
                self.policy.action_mean,
                self.policy.action_std,
                self.policy.entropy,
            )

        teacher_results = forward_group(teacher_indices, is_teacher=True)
        student_results = forward_group(student_indices, is_teacher=False)

        results = []

        for teacher_result, student_result in zip(teacher_results, student_results):
            combined = torch.cat((teacher_result, student_result), dim=0,)
            results.append(combined[restore_order])

        return tuple(results)

    def _update_learning_rate(
        self,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        new_mu: torch.Tensor,
        new_sigma: torch.Tensor,
    ) -> None:
        if self.desired_kl is None or self.schedule != "adaptive":
            return

        with torch.inference_mode():
            kl = torch.sum(torch.log(new_sigma/old_sigma + 1.0e-5)
            + (old_sigma.square() + (old_mu - new_mu).square()) / (2.0 * new_sigma.square()) - 0.5, dim=-1)
            kl_mean = kl.mean()

            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(
                    1.0e-5,
                    self.learning_rate / 1.5,
                    )
            elif 0.0 < kl_mean < self.desired_kl / 2.0:
                self.learning_rate = min(
                    1.0e-2,
                    self.learning_rate * 1.5,
                )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_student_losses: dict[str, float] = {}

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
        )

        for batch in generator:
            (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            history_batch,
            is_teacher_batch,
            ) = batch
            if self.normalize_advantage_per_mini_batch:
                advantages_batch = (
                    advantages_batch - advantages_batch.mean()
                ) / (advantages_batch.std() + 1.0e-8)

            (
                actions_log_prob_batch,
                value_batch,
                mu_batch,
                sigma_batch,
                entropy_batch,
            ) = self._forward_batch(obs_batch, history_batch, actions_batch, is_teacher_batch,)

            self._update_learning_rate(old_mu_batch, old_sigma_batch, mu_batch, sigma_batch,)

            old_log_prob_batch = old_log_prob_batch.squeeze(-1)
            advantages_batch = advantages_batch.squeeze(-1)

            ratio = torch.exp(
                actions_log_prob_batch - old_log_prob_batch
            )

            surrogate = -advantages_batch * ratio
            surrogate_clipped = (
                -advantages_batch
                * torch.clamp(
                    ratio,
                    1.0 - self.clip_param,
                    1.0 + self.clip_param,
                )
            )
            surrogate_losses = torch.maximum(surrogate, surrogate_clipped,)

            teacher_surrogate_loss = surrogate_losses[is_teacher_batch].mean()
            student_surrogate_loss = surrogate_losses[~is_teacher_batch].mean()

            surrogate_loss = teacher_surrogate_loss + student_surrogate_loss

            if self.use_clipped_value_loss:
               value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param,)

               value_losses = (value_batch - returns_batch).square()
               value_lossed_clipped = (value_clipped - returns_batch).square()

               value_loss = torch.maximum(value_losses, value_lossed_clipped).mean()
            else:
               value_loss = (value_batch - returns_batch).square().mean()

            entropy = entropy_batch.mean()

            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            ppo_loss.backward()
            nn.utils.clip_grad_norm_(self.ppo_parameters, self.max_grad_norm)
            self.optimizer.step()

            if self.policy.noise_std_type == "scalar":
               with torch.no_grad():
                   self.policy.std.clamp_(min=1.0e-6, max=10.0,)

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.item()

        latent_generator = self.storage.mini_batch_generator(self.num_mini_batches,self.num_learning_epochs,)

        for batch in latent_generator:
            (
                obs_batch,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                history_batch,
                is_teacher_batch,
            ) = batch

            student_mask = ~is_teacher_batch
            student_obs = obs_batch[student_mask]
            student_history = history_batch[student_mask]

            student_latent = self.policy.get_student_latent(student_history)

            student_loss, student_loss_terms = (
                self._compute_student_loss(student_obs, student_history,)
            )

            self.student_optimizer.zero_grad()

            student_loss.backward()
            nn.utils.clip_grad_norm_(
                    self.student_encoder_module.parameters(),
                    self.max_grad_norm,
                )

            self.student_optimizer.step()

            for name, loss_term in student_loss_terms.items():
                mean_student_losses[name] = mean_student_losses.get(name, 0.0) + loss_term.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }

        loss_dict.update({
            name: total / num_updates
            for name, total in mean_student_losses.items()}
            )

        return loss_dict
