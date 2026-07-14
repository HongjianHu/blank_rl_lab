from __future__ import annotations

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator
from tensordict import TensorDict

from rsl_rl.modules import ActorCriticCTS
from rsl_rl.utils import export_cts_policy_as_jit, export_cts_policy_as_onnx


NUM_ENVS = 3
NUM_ACTOR_OBS = 6
NUM_CRITIC_OBS = 10
NUM_ACTIONS = 3
HISTORY_LENGTH = 3


def make_policy() -> ActorCriticCTS:
    initial_obs = TensorDict(
        {
            "policy": torch.zeros(NUM_ENVS, NUM_ACTOR_OBS),
            "critic": torch.zeros(NUM_ENVS, NUM_CRITIC_OBS),
        },
        batch_size=[NUM_ENVS],
    )
    policy = ActorCriticCTS(
        obs=initial_obs,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=NUM_ACTIONS,
        history_length=HISTORY_LENGTH,
        latent_dim=4,
        actor_obs_normalization=True,
        actor_hidden_dims=[16, 8],
        critic_hidden_dims=[16, 8],
        teacher_encoder_hidden_dims=[16, 8],
        student_encoder_hidden_dims=[16, 8],
    ).eval()

    with torch.no_grad():
        policy.actor_obs_normalizer._mean.copy_(  # type: ignore[attr-defined]
            torch.linspace(-0.3, 0.2, NUM_ACTOR_OBS).unsqueeze(0)
        )
        policy.actor_obs_normalizer._std.copy_(  # type: ignore[attr-defined]
            torch.linspace(0.7, 1.3, NUM_ACTOR_OBS).unsqueeze(0)
        )
    return policy


def make_sequence(num_steps: int = 5) -> list[torch.Tensor]:
    base = torch.arange(NUM_ENVS * NUM_ACTOR_OBS, dtype=torch.float32).reshape(NUM_ENVS, NUM_ACTOR_OBS)
    return [base * 0.05 + step * 0.17 for step in range(num_steps)]


def as_tensordict(policy_obs: torch.Tensor) -> TensorDict:
    return TensorDict(
        {
            "policy": policy_obs,
            "critic": torch.zeros(NUM_ENVS, NUM_CRITIC_OBS),
        },
        batch_size=[NUM_ENVS],
    )


def test_jit_matches_multistep_student_inference_and_reset(tmp_path) -> None:
    torch.manual_seed(7)
    policy = make_policy()
    export_cts_policy_as_jit(policy, str(tmp_path), num_envs=NUM_ENVS)
    jit_policy = torch.jit.load(str(tmp_path / "policy.pt"), map_location="cpu")

    for obs in make_sequence():
        expected = policy.act_inference(as_tensordict(obs))
        actual = jit_policy(obs)
        torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)
        torch.testing.assert_close(jit_policy.history, policy._inference_history)  # type: ignore[attr-defined]

    policy.reset()
    jit_policy.reset()
    assert torch.count_nonzero(policy._inference_history) == 0  # type: ignore[attr-defined]
    assert torch.count_nonzero(jit_policy.history) == 0

    obs_after_reset = make_sequence(1)[0] + 1.25
    expected = policy.act_inference(as_tensordict(obs_after_reset))
    actual = jit_policy(obs_after_reset)
    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    torch.testing.assert_close(jit_policy.history, policy._inference_history)  # type: ignore[attr-defined]


def test_onnx_matches_multistep_student_inference_and_external_reset(tmp_path) -> None:
    torch.manual_seed(11)
    policy = make_policy()
    export_cts_policy_as_onnx(policy, str(tmp_path))

    model = onnx.load(tmp_path / "policy.onnx")
    onnx.checker.check_model(model)
    evaluator = ReferenceEvaluator(model)
    history = np.zeros((NUM_ENVS, HISTORY_LENGTH, NUM_ACTOR_OBS), dtype=np.float32)

    for obs in make_sequence():
        expected = policy.act_inference(as_tensordict(obs)).detach().numpy()
        actions, next_history = evaluator.run(
            None,
            {"obs": obs.numpy(), "history": history},
        )
        np.testing.assert_allclose(actions, expected, rtol=1.0e-5, atol=1.0e-6)
        np.testing.assert_allclose(
            next_history,
            policy._inference_history.detach().numpy(),  # type: ignore[attr-defined]
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        history = next_history

    policy.reset()
    history.fill(0.0)
    obs_after_reset = make_sequence(1)[0] + 1.25
    expected = policy.act_inference(as_tensordict(obs_after_reset)).detach().numpy()
    actions, next_history = evaluator.run(
        None,
        {"obs": obs_after_reset.numpy(), "history": history},
    )
    np.testing.assert_allclose(actions, expected, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(
        next_history,
        policy._inference_history.detach().numpy(),  # type: ignore[attr-defined]
        rtol=1.0e-6,
        atol=1.0e-6,
    )
