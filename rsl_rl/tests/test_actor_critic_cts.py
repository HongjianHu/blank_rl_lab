from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.modules.actor_critic_cts import ActorCriticCTS


NUM_ENVS = 8
NUM_ACTOR_OBS = 45
NUM_CRITIC_OBS = 263
NUM_ACTIONS = 12
HISTORY_LENGTH = 5
LATENT_DIM = 32


def make_obs(policy_value: float | None = None) -> TensorDict:
    policy = torch.randn(NUM_ENVS, NUM_ACTOR_OBS)
    if policy_value is not None:
        policy.fill_(policy_value)
    return TensorDict(
        {
            "policy": policy,
            "critic": torch.randn(NUM_ENVS, NUM_CRITIC_OBS),
        },
        batch_size=[NUM_ENVS],
    )


def make_model() -> ActorCriticCTS:
    return ActorCriticCTS(
        obs=make_obs(),
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=NUM_ACTIONS,
        history_length=HISTORY_LENGTH,
        latent_dim=LATENT_DIM,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        teacher_encoder_hidden_dims=[32, 16],
        student_encoder_hidden_dims=[32, 16],
    )


def has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in module.parameters()
    )


def test_shapes_and_distribution_interface() -> None:
    torch.manual_seed(0)
    model = make_model()
    obs = make_obs()
    history = torch.randn(NUM_ENVS, HISTORY_LENGTH, NUM_ACTOR_OBS)

    teacher_latent = model.get_teacher_latent(obs)
    student_latent = model.get_student_latent(history)
    assert teacher_latent.shape == (NUM_ENVS, LATENT_DIM)
    assert student_latent.shape == (NUM_ENVS, LATENT_DIM)
    torch.testing.assert_close(teacher_latent.norm(dim=-1), torch.ones(NUM_ENVS))
    torch.testing.assert_close(student_latent.norm(dim=-1), torch.ones(NUM_ENVS))

    teacher_actions = model.act(obs, history, is_teacher=True)
    assert teacher_actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert model.action_mean.shape == (NUM_ENVS, NUM_ACTIONS)
    assert model.action_std.shape == (NUM_ENVS, NUM_ACTIONS)
    assert model.entropy.shape == (NUM_ENVS,)
    assert model.get_actions_log_prob(teacher_actions).shape == (NUM_ENVS,)
    assert model.evaluate(obs, history, is_teacher=True).shape == (NUM_ENVS, 1)

    flat_history = history.flatten(start_dim=1)
    torch.testing.assert_close(
        model.get_student_latent(history),
        model.get_student_latent(flat_history),
    )
    assert model.act(obs, flat_history, is_teacher=False).shape == (NUM_ENVS, NUM_ACTIONS)
    assert model.evaluate(obs, flat_history, is_teacher=False).shape == (NUM_ENVS, 1)


def test_gradient_isolation_matches_cts_design() -> None:
    torch.manual_seed(1)
    model = make_model()
    obs = make_obs()
    history = torch.randn(NUM_ENVS, HISTORY_LENGTH, NUM_ACTOR_OBS)

    model.zero_grad(set_to_none=True)
    model.act(obs, history, is_teacher=True)
    model.action_mean.square().mean().backward()
    assert has_nonzero_grad(model.teacher_encoder)
    assert not has_nonzero_grad(model.student_encoder)

    model.zero_grad(set_to_none=True)
    model.act(obs, history, is_teacher=False)
    model.action_mean.square().mean().backward()
    assert not has_nonzero_grad(model.student_encoder)
    assert has_nonzero_grad(model.actor)

    model.zero_grad(set_to_none=True)
    model.evaluate(obs, history, is_teacher=True).square().mean().backward()
    assert not has_nonzero_grad(model.teacher_encoder)
    assert has_nonzero_grad(model.critic)

    model.zero_grad(set_to_none=True)
    teacher_target = model.get_teacher_latent(obs).detach()
    student_latent = model.get_student_latent(history)
    torch.nn.functional.mse_loss(student_latent, teacher_target).backward()
    assert has_nonzero_grad(model.student_encoder)
    assert not has_nonzero_grad(model.teacher_encoder)


def test_inference_history_and_reset() -> None:
    model = make_model()
    first_obs = make_obs(policy_value=1.0)
    second_obs = make_obs(policy_value=2.0)

    assert model.act_inference(first_obs).shape == (NUM_ENVS, NUM_ACTIONS)
    assert model._inference_history.shape == (NUM_ENVS, HISTORY_LENGTH, NUM_ACTOR_OBS) # type:ignore
    torch.testing.assert_close(model._inference_history[:, -1], first_obs["policy"])   # type:ignore

    model.act_inference(second_obs)
    torch.testing.assert_close(model._inference_history[:, -2], first_obs["policy"])   # type:ignore
    torch.testing.assert_close(model._inference_history[:, -1], second_obs["policy"])  # type:ignore

    dones = torch.zeros(NUM_ENVS, dtype=torch.bool)
    dones[0] = True
    model.reset(dones)
    assert torch.count_nonzero(model._inference_history[0]).item() == 0                # type:ignore
    assert torch.count_nonzero(model._inference_history[1]).item() > 0                 # type:ignore
    assert "_inference_history" not in model.state_dict()
