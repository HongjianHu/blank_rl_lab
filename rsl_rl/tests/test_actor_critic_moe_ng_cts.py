from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.modules import ActorCriticMoENGCTS


NUM_ENVS = 8
NUM_ACTOR_OBS = 45
NUM_CRITIC_OBS = 263
NUM_ACTIONS = 12
HISTORY_LENGTH = 5
LATENT_DIM = 32
EXPERT_NUM = 8
OBS_NO_GOAL_MASK = [True] * 6 + [False] * 3 + [True] * 36


def make_obs(value: float | None = None) -> TensorDict:
    policy = torch.randn(NUM_ENVS, NUM_ACTOR_OBS)
    if value is not None:
        policy.fill_(value)

    return TensorDict(
        {
            "policy": policy,
            "critic": torch.randn(NUM_ENVS, NUM_CRITIC_OBS),
        },
        batch_size=[NUM_ENVS],
    )


def make_model(mask: list[bool] = OBS_NO_GOAL_MASK) -> ActorCriticMoENGCTS:
    return ActorCriticMoENGCTS(
        obs=make_obs(),
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=NUM_ACTIONS,
        obs_no_goal_mask=mask,
        history_length=HISTORY_LENGTH,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        teacher_encoder_hidden_dims=[32, 16],
        student_encoder_hidden_dims=[32, 16],
        student_expert_num=EXPERT_NUM,
        student_expert_hidden_dim=16,
        latent_dim=LATENT_DIM,
    )


def has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in module.parameters()
    )


def test_student_shapes_and_framewise_no_goal_mask() -> None:
    torch.manual_seed(0)
    model = make_model()
    history = torch.arange(
        NUM_ENVS * HISTORY_LENGTH * NUM_ACTOR_OBS,
        dtype=torch.float32,
    ).reshape(NUM_ENVS, HISTORY_LENGTH, NUM_ACTOR_OBS)
    captured: dict[str, torch.Tensor] = {}

    def capture_moe_inputs(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        captured["full"] = inputs[0].detach().clone()
        captured["no_goal"] = inputs[1].detach().clone()

    hook = model.student_moe_encoder.register_forward_pre_hook(capture_moe_inputs)
    latent, weights = model.get_student_latent_and_weights(history)
    hook.remove()

    expected_full = history.flatten(start_dim=1)
    expected_no_goal = history[..., model.obs_no_goal_mask].flatten(start_dim=1)  # type: ignore

    assert latent.shape == (NUM_ENVS, LATENT_DIM)
    assert weights.shape == (NUM_ENVS, EXPERT_NUM)
    assert captured["full"].shape == (NUM_ENVS, 225)
    assert captured["no_goal"].shape == (NUM_ENVS, 210)
    torch.testing.assert_close(captured["full"], expected_full)
    torch.testing.assert_close(captured["no_goal"], expected_no_goal)
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones(NUM_ENVS),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        latent.norm(dim=-1),
        torch.ones(NUM_ENVS),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    assert not hasattr(model, "student_encoder")
    assert "obs_no_goal_mask" not in model.state_dict()


def test_teacher_student_interfaces_and_gradient_isolation() -> None:
    torch.manual_seed(1)
    model = make_model()
    obs = make_obs()
    history = torch.randn(NUM_ENVS, HISTORY_LENGTH, NUM_ACTOR_OBS)

    model.zero_grad(set_to_none=True)
    teacher_actions = model.act(obs, history, is_teacher=True)
    model.action_mean.square().mean().backward()
    assert teacher_actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert has_nonzero_grad(model.teacher_encoder)
    assert not has_nonzero_grad(model.student_moe_encoder)

    model.zero_grad(set_to_none=True)
    student_actions = model.act(obs, history, is_teacher=False)
    model.action_mean.square().mean().backward()
    assert student_actions.shape == (NUM_ENVS, NUM_ACTIONS)
    assert has_nonzero_grad(model.actor)
    assert not has_nonzero_grad(model.student_moe_encoder)

    model.zero_grad(set_to_none=True)
    teacher_target = model.get_teacher_latent(obs).detach()
    student_latent = model.get_student_latent(history)
    torch.nn.functional.mse_loss(student_latent, teacher_target).backward()
    assert has_nonzero_grad(model.student_moe_encoder)
    assert not has_nonzero_grad(model.teacher_encoder)

    assert model.evaluate(obs, history, is_teacher=True).shape == (NUM_ENVS, 1)
    assert model.evaluate(obs, history, is_teacher=False).shape == (NUM_ENVS, 1)


def test_inference_history_reset_and_mask_validation() -> None:
    model = make_model()
    first_obs = make_obs(value=1.0)
    second_obs = make_obs(value=2.0)

    assert model.act_inference(first_obs).shape == (NUM_ENVS, NUM_ACTIONS)
    torch.testing.assert_close(model._inference_history[:, -1], first_obs["policy"])  # type: ignore

    model.act_inference(second_obs)
    torch.testing.assert_close(model._inference_history[:, -2], first_obs["policy"])  # type: ignore
    torch.testing.assert_close(model._inference_history[:, -1], second_obs["policy"])  # type: ignore

    dones = torch.zeros(NUM_ENVS, dtype=torch.bool)
    dones[0] = True
    model.reset(dones)
    assert torch.count_nonzero(model._inference_history[0]).item() == 0  # type: ignore
    assert torch.count_nonzero(model._inference_history[1]).item() > 0  # type: ignore

    with pytest.raises(ValueError, match="policy observation"):
        make_model(mask=[True] * 44)
