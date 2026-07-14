from __future__ import annotations

from collections.abc import Generator, Sequence

import torch
from tensordict import TensorDict

from .rollout_storage import RolloutStorage

class RolloutStorageCTS(RolloutStorage):
    """Rollout storage for concurrent teacher-student PPO."""

    class Transition(RolloutStorage.Transition):
        def __init__(self) -> None:
           super().__init__()
           self.history: torch.Tensor | None = None

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        actions_shape: tuple[int, ...] | list[int],
        history_length: int,
        teacher_env_ids: Sequence[int] | torch.Tensor,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            training_type="rl",
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs=obs,
            actions_shape=actions_shape,
            device=device,
        )

        if history_length < 1:
           raise ValueError("history_length must be greater than zero.")

        if "policy" not in obs_groups:
           raise KeyError("obs_groups must contain a 'policy' group.")

        self.obs_groups = obs_groups
        self.history_length = history_length

        self.num_actor_obs = 0
        for group_name in obs_groups["policy"]:
            group_obs = obs[group_name]
            if len(group_obs.shape) != 2:
                raise ValueError(
                    "RolloutStorageCTS only supports vector policy "
                    f"observations, but '{group_name}' has shape "
                    f"{group_obs.shape}."
                )
            self.num_actor_obs += group_obs.shape[-1]

        teacher_env_ids = torch.as_tensor(
            teacher_env_ids,
            dtype=torch.long,
            device=device,
        ).flatten()

        if teacher_env_ids.numel() == 0:
           raise ValueError("CTS requires at least one teacher environment.")

        if torch.any(teacher_env_ids < 0) or torch.any(
            teacher_env_ids >= num_envs
        ):
          raise ValueError("teacher_env_ids contains an invalid environment ID.")

        if torch.unique(teacher_env_ids).numel() != teacher_env_ids.numel():
           raise ValueError("teacher_env_ids contains duplicate IDs.")

        teacher_mask = torch.zeros(
           num_envs,
           dtype=torch.bool,
           device=device,
        )

        teacher_mask[teacher_env_ids] = True

        self.teacher_env_ids = teacher_env_ids

        self.student_env_ids = torch.nonzero(
            ~teacher_mask,
            as_tuple=True)[0]

        if self.student_env_ids.numel() == 0:
           raise ValueError("CTS requires at least one student environment.")

        self.teacher_mask = teacher_mask
        self.num_teacher_envs = teacher_env_ids.numel()
        self.num_student_envs = self.student_env_ids.numel()

        self.history = torch.zeros(
           num_transitions_per_env,
           num_envs,
           history_length * self.num_actor_obs,
           device=device,
        )

    def _flatten_history(self, history: torch.Tensor) -> torch.Tensor:
        framed_shape = (
            self.num_envs,
            self.history_length,
            self.num_actor_obs,
        )
        flat_shape = (
            self.num_envs,
            self.history_length * self.num_actor_obs,
        )

        if tuple(history.shape) == framed_shape:
            return history.flatten(start_dim=1)

        if tuple(history.shape) == flat_shape:
            return history

        raise ValueError(
            f"Invalid history shape {tuple(history.shape)}. "
            f"Expected {framed_shape} or {flat_shape}."
        )

    def add_transition(self, transition: Transition) -> None:
        if transition.history is None:
           raise ValueError("CTS transition.history must be populated.")

        history = self._flatten_history(transition.history)
        storage_step = self.step

        super().add_transition(transition)
        self.history[storage_step].copy_(history)

    def _flat_indices_for_envs(self, env_ids: torch.Tensor,) -> torch.Tensor:
        time_offsets = (
            torch.arange(
            self.num_transitions_per_env,
            device=self.device,
        ) * self.num_envs
        )

        return (time_offsets[:, None] + env_ids[None, :]).flatten()

    def mini_batch_generator(self, num_mini_batches, num_epochs = 8) -> Generator:
        if num_mini_batches < 1:
           raise ValueError("num_mini_batches must be greater than zero.")

        teacher_indices = self._flat_indices_for_envs(self.teacher_env_ids)
        student_indices = self._flat_indices_for_envs(self.student_env_ids)

        teacher_batch_size = teacher_indices.numel() // num_mini_batches
        student_batch_size = student_indices.numel() // num_mini_batches

        if teacher_batch_size == 0:
            raise ValueError(
                "The number of teacher samples must be at least "
                "num_mini_batches."
            )

        if student_batch_size == 0:
            raise ValueError(
                "The number of student samples must be at least "
                "num_mini_batches."
            )

        observations = self.observations.flatten(0, 1) #[num_transitions_per_env, num_envs, 1]
        history = self.history.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            teacher_permutation = torch.randperm(
                teacher_indices.numel(),
                device=self.device,
            )
            student_permutation = torch.randperm(
                student_indices.numel(),
                device=self.device,
            )

            for batch_id in range(num_mini_batches):
                teacher_start = batch_id * teacher_batch_size
                teacher_stop = teacher_start + teacher_batch_size
                student_start = batch_id * student_batch_size
                student_stop = student_start + student_batch_size

                teacher_batch_indices = teacher_indices[
                    teacher_permutation[teacher_start:teacher_stop]
                ]

                student_batch_indices = student_indices[
                    student_permutation[student_start:student_stop]
                ]

                batch_indices = torch.cat(
                    (
                        teacher_batch_indices,
                        student_batch_indices,
                    ),
                    dim=0,
                )

                is_teacher_batch = torch.zeros(
                    batch_indices.numel(),
                    dtype=torch.bool,
                    device=self.device,
                )

                is_teacher_batch[:teacher_batch_size] = True

                yield(
                    observations[batch_indices], # type:ignore
                    actions[batch_indices],
                    values[batch_indices],
                    advantages[batch_indices],
                    returns[batch_indices],
                    old_actions_log_prob[batch_indices],
                    old_mu[batch_indices],
                    old_sigma[batch_indices],
                    history[batch_indices],
                    is_teacher_batch,
                )