from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.actor_critic_dwaq import ActorCriticDWAQ
from rsl_rl.modules.vae_context import ContextVAEOutput
from rsl_rl.storage import RolloutStorage

class DWAQPPO(PPO):

    policy: ActorCriticDWAQ

    def __init__(
        self,
        policy: ActorCriticDWAQ,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "fixed",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        # DWAQ-specific parameters
        beta: float = 1.0,
        vae_learning_rate: float = 1e-3,
    ) -> None:
        super().__init__(
            policy=policy,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            share_cnn_encoders=False,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        self.beta = beta

        rl_parameters = (
            list(self.policy.actor.parameters())
            + list(self.policy.critic.parameters())
            + [self.policy.std]
        )

        self.optimizer = optim.Adam(rl_parameters, lr=learning_rate)
        self.vae_optimizer = optim.Adam(
            self.policy.context_vae.parameters(), lr=vae_learning_rate
        )

    def update(self) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_autoenc_loss = 0.0
        mean_rnd_loss = 0 if self.rnd else None
        mean_symmetry_loss = 0 if self.symmetry else None

        # Pre-compute o_{t+1} for VAE reconstruction target
        T = self.storage.num_transitions_per_env
        N = self.storage.num_envs

        policy_keys = self.policy.obs_groups["policy"]
        policy_obs_seq = torch.cat(
            [self.storage.observations[k] for k in policy_keys], dim=-1
        ) # [T, N, policy_dim]

        self.storage.observations["_next_policy_obs"] = torch.cat(
            [policy_obs_seq[1:], policy_obs_seq[-1:]], dim=0
        )

        live_mask = 1.0 - self.storage.dones.float()  # [T, N, 1]
        self.storage.observations["_live_mask"] = live_mask

        # recon_mask: live_mask + last step also masked (no valid o_{t+1})
        recon_mask = live_mask.clone()
        recon_mask[-1] = 0.0
        self.storage.observations["_recon_mask"] = recon_mask

        # Mini-batch generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for ( # M = total_samples // num_mini_batches，
            obs_batch, # [M, obs_dim]
            actions_batch,
            target_values_batch,
            advantages_batch, # [M, 1]
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1 
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
               with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # 父类 self.symmetry = symmetry_cfg
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"]
                )
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # hidden_states_batch[0] actor循环神经网络（RNN / LSTM / GRU）在时间序列起点处的隐藏状态
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0]) # 考虑当存在循环神经网络的情况  
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum( # type:ignore
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        # 将不同卡各自算出的kl_mean相加，再除以总卡数。此时，每张卡内存中的kl_mean都变成了全局统一的平均KL值。
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        # 将Rank 0算出的新学习率张量，广播（发送）给所有其他GPU进程。其他Rank接收到后，更新本地的self.learning_rate
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            rl_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    num_aug = int(obs_batch.shape[0] / original_batch_size)
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                symmetry_loss = F.mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    rl_loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()
            
            self.optimizer.zero_grad()
            rl_loss.backward()
            nn.utils.clip_grad_norm_(
                [p for group in self.optimizer.param_groups for p in group["params"]],
                self.max_grad_norm,
            )
            self.optimizer.step()

            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                rnd_loss = F.mse_loss(predicted_embedding, target_embedding)
                self.rnd_optimizer.zero_grad() # type:ignore
                rnd_loss.backward()
                self.rnd_optimizer.step() # type:ignore

            obs_history = self.policy.get_obs_history(obs_batch)
            vae_out: ContextVAEOutput = self.policy.cenet_forward(obs_history)

            vel_target = self.policy.get_velocity(obs_batch).detach()
            decode_target = obs_batch["_next_policy_obs"].detach()

            live = obs_batch["_live_mask"].detach()
            recon_m = obs_batch["_recon_mask"].detach()

            # Velocity MSE — masked by live (skip terminated envs)
            vel_loss = F.mse_loss(vae_out.code_vel * live, vel_target * live) # 同步监督信号

            # Reconstruction MSE — masked by recon_mask (skip terminated + last step)
            recon_loss = F.mse_loss(vae_out.reconstruction * recon_m, decode_target * recon_m)

            kl_divergence = -0.5 * torch.mean(
                torch.sum(1 + vae_out.logvar_latent - vae_out.mean_latent.pow(2) - vae_out.logvar_latent.exp(), dim=-1)
                * live.squeeze(-1)
            )

            autoenc_loss = vel_loss + recon_loss + self.beta * kl_divergence

            self.vae_optimizer.zero_grad()
            autoenc_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.context_vae.parameters(), self.max_grad_norm)
            self.vae_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_autoenc_loss += autoenc_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches # 本轮数据中网络参数被梯度更新的总次数
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_autoenc_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
             mean_symmetry_loss /= num_updates

        del self.storage.observations["_next_policy_obs"]
        del self.storage.observations["_live_mask"]
        del self.storage.observations["_recon_mask"]
        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "autoencoder": mean_autoenc_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict       





