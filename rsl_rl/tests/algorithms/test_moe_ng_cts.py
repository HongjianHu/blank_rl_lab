from __future__ import annotations

import math

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import MoENGCTS
from rsl_rl.modules import ActorCriticMoENGCTS
from rsl_rl.storage import RolloutStorageCTS


NUM_ENVS = 8
NUM_STEPS = 4
POLICY_DIM = 6
CRITIC_DIM = 10
NUM_ACTIONS = 3
HISTORY_LENGTH = 3
LATENT_DIM = 8
EXPERT_NUM = 4
TEACHER_ENV_IDS = torch.tensor([0, 1, 2, 4, 5, 6])
NO_GOAL_MASK = [True, True, False, False, True, True]


def make_obs() -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(NUM_ENVS, POLICY_DIM),
            "critic": torch.randn(NUM_ENVS, CRITIC_DIM),
        },
        batch_size=[NUM_ENVS],
    )


def make_algorithm(
    load_balance_coef: float = 0.25,
) -> tuple[MoENGCTS, TensorDict, torch.Tensor]:
    obs = make_obs()
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = ActorCriticMoENGCTS(
        obs=obs,
        obs_groups=obs_groups,
        num_actions=NUM_ACTIONS,
        obs_no_goal_mask=NO_GOAL_MASK,
        history_length=HISTORY_LENGTH,
        latent_dim=LATENT_DIM,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        teacher_encoder_hidden_dims=[32, 16],
        student_encoder_hidden_dims=[32, 16],
        student_expert_num=EXPERT_NUM,
        student_expert_hidden_dim=8,
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
    algorithm = MoENGCTS(
        policy=policy,
        storage=storage,
        load_balance_coef=load_balance_coef,
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


def clone_parameters(
    parameters: list[torch.nn.Parameter],
) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in parameters]


def any_parameter_changed(
    before: list[torch.Tensor],
    parameters: list[torch.nn.Parameter],
) -> bool:
    return any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, parameters)
    )


def optimizer_parameter_ids(
    optimizer: torch.optim.Optimizer,
) -> set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def test_student_loss_matches_latent_plus_load_balance() -> None:
    torch.manual_seed(3)
    coefficient = 0.25
    algorithm, obs, history = make_algorithm(coefficient)
    student_ids = algorithm.student_env_ids
    student_obs = obs[student_ids]
    student_history = history[student_ids]

    total_loss, loss_terms = algorithm._compute_student_loss(
        student_obs,
        student_history,
    )

    assert set(loss_terms) == {"latent", "load_balance"}
    torch.testing.assert_close(
        total_loss,
        loss_terms["latent"]
        + coefficient * loss_terms["load_balance"],
    )
    assert loss_terms["latent"].item() >= 0.0
    assert loss_terms["load_balance"].item() >= 0.0

    with pytest.raises(ValueError, match="non-negative"):
        make_algorithm(load_balance_coef=-0.01)


def test_full_rollout_and_update_changes_ppo_gate_and_experts() -> None:
    torch.manual_seed(7)
    algorithm, obs, history = make_algorithm()

    ppo_parameters = list(algorithm.ppo_parameters)
    student_parameters = list(
        algorithm.policy.student_moe_encoder.parameters()
    )
    gate_parameters = list(
        algorithm.policy.student_moe_encoder.gating_network.parameters()
    )
    expert_parameters = [
        *algorithm.policy.student_moe_encoder.experts_backbone.parameters(),
        *algorithm.policy.student_moe_encoder.experts_hidden.parameters(),
        *algorithm.policy.student_moe_encoder.experts_out.parameters(),
    ]

    ppo_ids = {id(parameter) for parameter in ppo_parameters}
    student_ids = {id(parameter) for parameter in student_parameters}
    assert ppo_ids.isdisjoint(student_ids)
    assert optimizer_parameter_ids(algorithm.optimizer) == ppo_ids
    assert optimizer_parameter_ids(algorithm.student_optimizer) == student_ids
    assert algorithm.student_encoder_module is algorithm.policy.student_moe_encoder

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
        history = torch.cat(
            (history[:, 1:], next_obs["policy"].unsqueeze(1)),
            dim=1,
        )
        obs = next_obs

    assert algorithm.storage.step == NUM_STEPS
    algorithm.compute_returns(obs, history)
    assert torch.isfinite(algorithm.storage.returns).all()
    assert torch.isfinite(algorithm.storage.advantages).all()

    ppo_before = clone_parameters(ppo_parameters)
    gate_before = clone_parameters(gate_parameters)
    expert_before = clone_parameters(expert_parameters)

    loss_dict = algorithm.update()

    assert set(loss_dict) == {
        "value",
        "surrogate",
        "entropy",
        "latent",
        "load_balance",
    }
    assert all(math.isfinite(value) for value in loss_dict.values())
    assert loss_dict["latent"] >= 0.0
    assert loss_dict["load_balance"] >= 0.0
    assert any_parameter_changed(ppo_before, ppo_parameters)
    assert any_parameter_changed(gate_before, gate_parameters)
    assert any_parameter_changed(expert_before, expert_parameters)
    assert algorithm.optimizer.state
    assert algorithm.student_optimizer.state
    assert algorithm.storage.step == 0
