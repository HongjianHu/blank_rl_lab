from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.storage.rollout_storage_cts import RolloutStorageCTS


NUM_ENVS = 4
NUM_STEPS = 3
POLICY_DIM = 2
CRITIC_DIM = 3
ACTION_DIM = 1
HISTORY_LENGTH = 2
TEACHER_ENV_IDS = torch.tensor([0, 2])


def make_empty_obs() -> TensorDict:
    return TensorDict(
        {
            "policy": torch.zeros(NUM_ENVS, POLICY_DIM),
            "critic": torch.zeros(NUM_ENVS, CRITIC_DIM),
        },
        batch_size=[NUM_ENVS],
    )


def make_storage() -> RolloutStorageCTS:
    return RolloutStorageCTS(
        num_envs=NUM_ENVS,
        num_transitions_per_env=NUM_STEPS,
        obs=make_empty_obs(),
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        actions_shape=[ACTION_DIM],
        history_length=HISTORY_LENGTH,
        teacher_env_ids=TEACHER_ENV_IDS,
    )


def populate_storage(storage: RolloutStorageCTS) -> None:
    for step in range(NUM_STEPS):
        sample_ids = torch.arange(NUM_ENVS, dtype=torch.float32) + step * NUM_ENVS
        transition = RolloutStorageCTS.Transition()
        transition.observations = TensorDict(
            {
                "policy": sample_ids[:, None].repeat(1, POLICY_DIM),
                "critic": (sample_ids + 100)[:, None].repeat(1, CRITIC_DIM),
            },
            batch_size=[NUM_ENVS],
        )
        transition.actions = (sample_ids + 300)[:, None]
        transition.rewards = sample_ids + 350
        transition.dones = torch.zeros(NUM_ENVS, dtype=torch.bool)
        transition.values = (sample_ids + 400)[:, None]
        transition.actions_log_prob = sample_ids + 700
        transition.action_mean = (sample_ids + 800)[:, None]
        transition.action_sigma = (sample_ids + 900)[:, None]
        transition.history = (sample_ids + 200)[:, None, None].repeat(
            1, HISTORY_LENGTH, POLICY_DIM
        )
        storage.add_transition(transition)

    sample_ids = torch.arange(NUM_STEPS * NUM_ENVS, dtype=torch.float32).reshape(NUM_STEPS, NUM_ENVS)
    storage.advantages[..., 0].copy_(sample_ids + 500)
    storage.returns[..., 0].copy_(sample_ids + 600)


def test_teacher_and_student_flat_indices() -> None:
    storage = make_storage()

    torch.testing.assert_close(
        storage._flat_indices_for_envs(storage.teacher_env_ids),
        torch.tensor([0, 2, 4, 6, 8, 10]),
    )
    torch.testing.assert_close(
        storage._flat_indices_for_envs(storage.student_env_ids),
        torch.tensor([1, 3, 5, 7, 9, 11]),
    )


def test_mini_batches_preserve_identity_and_teacher_student_ratio() -> None:
    torch.manual_seed(0)
    storage = make_storage()
    populate_storage(storage)

    batches = list(storage.mini_batch_generator(num_mini_batches=2, num_epochs=2))
    assert len(batches) == 4

    ids_by_epoch: list[torch.Tensor] = []
    for batch_index, batch in enumerate(batches):
        (
            obs_batch,
            actions_batch,
            values_batch,
            advantages_batch,
            returns_batch,
            old_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            history_batch,
            is_teacher_batch,
        ) = batch

        sample_ids = obs_batch["policy"][:, 0]
        sample_env_ids = sample_ids.long() % NUM_ENVS

        assert batch_index % 2 in (0, 1)
        assert is_teacher_batch.sum().item() == 3
        assert (~is_teacher_batch).sum().item() == 3
        assert torch.all(torch.isin(sample_env_ids[is_teacher_batch], torch.tensor([0, 2])))
        assert torch.all(torch.isin(sample_env_ids[~is_teacher_batch], torch.tensor([1, 3])))

        torch.testing.assert_close(obs_batch["policy"], sample_ids[:, None].repeat(1, POLICY_DIM))
        torch.testing.assert_close(obs_batch["critic"], (sample_ids + 100)[:, None].repeat(1, CRITIC_DIM))
        torch.testing.assert_close(history_batch, (sample_ids + 200)[:, None].repeat(1, HISTORY_LENGTH * POLICY_DIM))
        torch.testing.assert_close(actions_batch[:, 0], sample_ids + 300)
        torch.testing.assert_close(values_batch[:, 0], sample_ids + 400)
        torch.testing.assert_close(advantages_batch[:, 0], sample_ids + 500)
        torch.testing.assert_close(returns_batch[:, 0], sample_ids + 600)
        torch.testing.assert_close(old_log_prob_batch[:, 0], sample_ids + 700)
        torch.testing.assert_close(old_mu_batch[:, 0], sample_ids + 800)
        torch.testing.assert_close(old_sigma_batch[:, 0], sample_ids + 900)

        if batch_index % 2 == 0:
            ids_by_epoch.append(sample_ids)
        else:
            ids_by_epoch[-1] = torch.cat((ids_by_epoch[-1], sample_ids))

    expected_ids = torch.arange(NUM_STEPS * NUM_ENVS, dtype=torch.float32)
    for epoch_ids in ids_by_epoch:
        torch.testing.assert_close(epoch_ids.sort().values, expected_ids)
