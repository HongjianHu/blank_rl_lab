# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import numpy as np
from typing import Optional, Tuple, List, Generator, Union

from rsl_rl.utils import split_and_pad_trajectories


class FlatRolloutStorage:
    """Storage for the data collected during a rollout.

    The rollout storage is populated by adding transitions during the rollout phase. It then returns a generator for
    learning, depending on the algorithm and the policy architecture.
    """

    class Transition:
        """Storage for a single state transition.

        This class is populated incrementally during the rollout phase and then passed to
        :meth:`RolloutStorage.add_transition` to record the data.
        """

        def __init__(self) -> None:
            """Initialize an empty transition container."""
            self.observations: Optional[torch.Tensor] | None = None
            self.critic_observations: Optional[torch.Tensor] = None
            self.actions: Optional[torch.Tensor] = None
            self.rewards: Optional[torch.Tensor] = None
            self.dones: Optional[torch.Tensor] = None
            self.values: Optional[torch.Tensor] = None
            self.actions_log_prob: Optional[torch.Tensor] = None
            self.action_mean: Optional[torch.Tensor] = None
            self.action_sigma: Optional[torch.Tensor] = None

            self.hidden_states: Optional[Tuple[torch.Tensor, ...]] = None

        def clear(self) -> None:
            """Reset all transition fields to None."""
            self.__init__()

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: Tuple[int, ...],
        privileged_obs_shape: Tuple[Optional[int], ...],
        actions_shape: Tuple[int, ...],
        device: str = "cpu",
    ) -> None:
        self.device: str = device
        self.obs_shape: Tuple[int, ...] = obs_shape
        self.privileged_obs_shape: Tuple[Optional[int], ...] = privileged_obs_shape
        self.actions_shape: Tuple[int, ...] = actions_shape
        self.observations: torch.Tensor = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.privileged_observations: Optional[torch.Tensor]
        if privileged_obs_shape[0] is not None:
           self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device) # type:ignore
        else:
           self.privileged_observations = None 
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        self.actions_log_prob: torch.Tensor = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values: torch.Tensor = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns: torch.Tensor = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages: torch.Tensor = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu: torch.Tensor = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.num_transitions_per_env: int = num_transitions_per_env
        self.num_envs: int = num_envs
        self.saved_hidden_states_a: Optional[List[torch.Tensor]] = None
        self.saved_hidden_states_c: Optional[List[torch.Tensor]] = None
        self.step: int = 0

    def add_transitions(self, transition: Transition) -> None:
        """Add one transition to the storage at the current step index."""
        # Check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")
        assert transition.observations is not None
        assert transition.actions is not None
        assert transition.rewards is not None
        assert transition.dones is not None
        assert transition.values is not None
        assert transition.actions_log_prob is not None
        assert transition.action_mean is not None
        assert transition.action_sigma is not None
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None and transition.critic_observations is not None:
           self.privileged_observations[self.step].copy_(transition.critic_observations) 
        self.actions[self.step].copy_(transition.actions)  # type: ignore
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1)) # type: ignore
        self.dones[self.step].copy_(transition.dones.view(-1, 1)) # type: ignore
        self.values[self.step].copy_(transition.values)     # type: ignore
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1)) #  type:ignore
        self.mu[self.step].copy_(transition.action_mean)     # type: ignore
        self.sigma[self.step].copy_(transition.action_sigma) # type: ignore            
        self._save_hidden_states(transition.hidden_states)   # type: ignore
        self.step += 1

    def clear(self) -> None:
        """Reset the write cursor for the next rollout."""
        self.step = 0


    def _save_hidden_states(self, hidden_states: Optional[Tuple[torch.Tensor, ...]]) -> None:
        if hidden_states is None or hidden_states == (None, None):
            return
        # Make a tuple out of GRU hidden states to match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        # Initialize hidden states if needed
        if self.saved_hidden_state_a is None: # type:ignore
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        assert self.saved_hidden_states_a is not None
        assert self.saved_hidden_states_c is not None

        # Copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_state_a[i][self.step].copy_(hidden_state_a[i])  #  type:ignore
            self.saved_hidden_state_c[i][self.step].copy_(hidden_state_c[i])  #  type:ignore

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float) -> None:
        # delta = self.rewards[step] + ... * gamma * next_values - self.values[step]
        # next_values 就是 V(s_{t+1})，V(s_{t+1}) = self.values[t+1] = self.values[step + 1]
        # rollout 存的是 values[0] ~ values[T-1]，没有 values[T]。所以最后一个 transition（step = T-1）的 V(s_T) 需要额外传入——这就是last_values
        # 是 rollout 结束后调用 Critic 对最终观测评估一次得到的。
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-08)

    def get_statistics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # self.dones[self.step].copy_(transition.dones.view(-1, 1))
        # self.dones.shape  # (T, N, 1)   T=每环境的transition数, N=并行环境数
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1) # (T, N, 1) → (N, T, 1) → (N*T, 1)
        # 展平后的顺序是 env0 的全部时间步 → env1 的全部时间步 -> ...
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        # flat_dones.nonzero() → [2, 6, 9, 14, 19]
        # done_indices = [-1, 2, 6, 9, 14, 19]
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return (trajectory_lengths.float().mean(), self.rewards.mean())
    
    # For reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations
        # Flatten the data
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                # Yield the mini-batch
                obs_batch = observations[batch_idx] #    type:ignore
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]        # type:ignore
                old_sigma_batch = old_sigma[batch_idx]  # type:ignore
                            # Yield the mini-batch
                yield (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (
                        None,
                        None,
                    ),
                    None,
                )
    # For reinforcement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            (padded_critic_obs_trajectories, _) = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_critic_obs_trajectories = padded_obs_trajectories
        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size
                # trajectory_masks: [max_traj_len, num_trajectories]  True=真实数据
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                # padded_obs_trajectories: [max_traj_len, batch_traj_num, obs_dim]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]
                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # Reshape to [num_envs, time, num layers, hidden dim]
                # Original shape: [time, num_layers, num_envs, hidden_dim])
                last_was_done = last_was_done.permute(1, 0) #  转置后 shape: [num_envs, num_steps]
                assert self.saved_hidden_states_a is not None
                assert self.saved_hidden_states_c is not None
                # Take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                hid_a_batch = [
                    # shape: [num_steps, num_layers, num_envs, hidden_dim] → [num_envs, num_steps, num_layers, hidden_dim]
                    # [last_was_done] 取出所有轨迹起点的隐状态 -> # → [num_trajectories_total, num_layers, hidden_dim]
                    # [first_traj:last_traj] 取这个 mini-batch 的部分 -> # → [num_layers, batch_traj_num, hidden_dim]
                    saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_state in self.saved_hidden_state_a # type:ignore
                ]
                hid_c_batchh = [
                    saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_state in self.saved_hidden_state_c  #type:ignore
                ]
                # Remove the tuple for GRU
                # GRU 兼容处理（GRU 只有 h，没有 c)
                hid_a_batch = (
                    hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                )
                hid_c_batchh = (
                    hid_c_batchh[0] if len(hid_c_batchh) == 1 else hid_c_batchh # LSTM：保留 (h, c) 格式
                )

                # Yield the mini-batch
                yield (
                    obs_batch,
                    actions_batch,
                    values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (
                        hid_a_batch,
                        hid_c_batchh,
                    ),
                    masks_batch,
                )

                first_traj = last_traj
