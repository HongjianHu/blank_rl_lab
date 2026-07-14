from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.cts import CTS
from rsl_rl.modules.actor_critic_cts import ActorCriticCTS
from rsl_rl.storage.rollout_storage_cts import RolloutStorageCTS


NUM_ENVS = 8
NUM_STEPS = 4
POLICY_DIM = 6
CRITIC_DIM = 10
NUM_ACTIONS = 3
HISTORY_LENGTH = 3
TEACHER_ENV_IDS = torch.tensor([0, 1, 2, 4, 5, 6])


def make_obs() -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(NUM_ENVS, POLICY_DIM),
            "critic": torch.randn(NUM_ENVS, CRITIC_DIM),
        },
        batch_size=[NUM_ENVS],
    )


def make_algorithm() -> tuple[CTS, TensorDict, torch.Tensor]:
    obs = make_obs()
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = ActorCriticCTS(
        obs=obs,
        obs_groups=obs_groups,
        num_actions=NUM_ACTIONS,
        history_length=HISTORY_LENGTH,
        latent_dim=8,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        teacher_encoder_hidden_dims=[32, 16],
        student_encoder_hidden_dims=[32, 16],
    )
    storage = RolloutStorageCTS(
        num_envs=NUM_ENVS,
        num_transitions_per_env=NUM_STEPS,
        obs=obs,
        obs_groups=obs_groups,
        actions_shape=[NUM_ACTIONS],
        history_length=HISTORY_LENGTH,
        teacher_env_ids=TEACHER_ENV_IDS,
    )
    algorithm = CTS(
        policy=policy,
        storage=storage,
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=3.0e-3,
        student_encoder_learning_rate=3.0e-3,
        teacher_env_ratio=0.75,
        schedule="fixed",
    )

    history = torch.zeros(NUM_ENVS, HISTORY_LENGTH, POLICY_DIM)
    history[:, -1] = obs["policy"]
    return algorithm, obs, history


def clone_parameters(parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def any_parameter_changed(
    before: list[torch.Tensor],
    parameters: list[torch.nn.Parameter],
) -> bool:
    return any(not torch.equal(old, new.detach()) for old, new in zip(before, parameters))


def test_full_cts_rollout_and_update_changes_both_optimizer_groups() -> None:
    torch.manual_seed(7)
    algorithm, obs, history = make_algorithm()

    primary_ids = {id(parameter) for parameter in algorithm.ppo_parameters}
    student_parameters = list(algorithm.policy.student_encoder.parameters())
    student_ids = {id(parameter) for parameter in student_parameters}
    assert primary_ids.isdisjoint(student_ids)

    for step in range(NUM_STEPS):
        actions = algorithm.act(obs, history)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)

        next_obs = make_obs()
        rewards = 1.0 - 0.05 * actions.square().sum(dim=-1)
        rewards += 0.01 * next_obs["policy"][:, 0]
        dones = torch.zeros(NUM_ENVS, dtype=torch.bool)
        time_outs = torch.zeros(NUM_ENVS)
        if step == 2:
            dones[0] = True
            time_outs[0] = 1.0

        algorithm.process_env_step(
            next_obs,
            rewards,
            dones,
            {"time_outs": time_outs},
        )

        history[dones] = 0.0
        history = torch.cat((history[:, 1:], next_obs["policy"].unsqueeze(1)), dim=1)
        obs = next_obs

    assert algorithm.storage.step == NUM_STEPS
    algorithm.compute_returns(obs, history)
    assert torch.isfinite(algorithm.storage.returns).all()
    assert torch.isfinite(algorithm.storage.advantages).all()
    assert abs(algorithm.storage.advantages.mean().item()) < 1.0e-5

    primary_before = clone_parameters(algorithm.ppo_parameters)
    student_before = clone_parameters(student_parameters)

    loss_dict = algorithm.update()

    assert set(loss_dict) == {"value", "surrogate", "entropy", "latent"}
    assert all(math.isfinite(value) for value in loss_dict.values())
    assert any_parameter_changed(primary_before, algorithm.ppo_parameters)
    assert any_parameter_changed(student_before, student_parameters)
    assert len(algorithm.optimizer.state) > 0
    assert len(algorithm.student_optimizer.state) > 0
    assert algorithm.storage.step == 0
