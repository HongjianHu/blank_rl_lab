from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunnerCTS


NUM_ENVS = 8
NUM_ACTIONS = 3
POLICY_DIM = 6
CRITIC_DIM = 10
HISTORY_LENGTH = 3
NUM_STEPS = 4


class FakeVecEnv(VecEnv):
    """Deterministic VecEnv for testing CTS without launching Isaac Sim."""

    def __init__(self, device: str = "cpu") -> None:
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = 100
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {}
        self.step_count = 0

    def observations_at(self, step: int) -> TensorDict:
        env_ids = torch.arange(self.num_envs, dtype=torch.float32, device=self.device).unsqueeze(1)
        policy = 100.0 * step + 10.0 * env_ids + torch.arange(POLICY_DIM, device=self.device)
        critic = 1000.0 + 100.0 * step + 10.0 * env_ids + torch.arange(CRITIC_DIM, device=self.device)
        return TensorDict(
            {"policy": policy, "critic": critic},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def get_observations(self) -> TensorDict:
        return self.observations_at(self.step_count)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        assert actions.shape == (self.num_envs, self.num_actions)
        self.step_count += 1
        self.episode_length_buf += 1

        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.step_count == 2:
            dones[0] = True
            self.episode_length_buf[0] = 0

        obs = self.get_observations()
        rewards = 1.0 - 0.01 * actions.square().sum(dim=-1)
        extras = {"time_outs": torch.zeros(self.num_envs, device=self.device)}
        return obs, rewards, dones, extras


def make_train_cfg() -> dict[str, Any]:
    return {
        "num_steps_per_env": NUM_STEPS,
        "save_interval": 100,
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "policy": {
            "class_name": "ActorCriticCTS",
            "history_length": HISTORY_LENGTH,
            "latent_dim": 8,
            "actor_hidden_dims": [32, 16],
            "critic_hidden_dims": [32, 16],
            "teacher_encoder_hidden_dims": [32, 16],
            "student_encoder_hidden_dims": [32, 16],
            "init_noise_std": 0.5,
        },
        "algorithm": {
            "class_name": "CTS",
            "num_learning_epochs": 1,
            "num_mini_batches": 2,
            "learning_rate": 3.0e-3,
            "student_encoder_learning_rate": 3.0e-3,
            "teacher_env_ratio": 0.75,
            "schedule": "fixed",
            "rnd_cfg": None,
        },
        "logger": "tensorboard",
    }


def make_runner() -> OnPolicyRunnerCTS:
    return OnPolicyRunnerCTS(FakeVecEnv(), make_train_cfg(), log_dir=None, device="cpu")


def assert_nested_equal(actual: Any, expected: Any, path: str = "root") -> None:
    """Compare model and optimizer state dictionaries with exact tensor equality."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        assert torch.equal(actual, expected), path
    elif isinstance(expected, Mapping):
        assert isinstance(actual, Mapping), path
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_nested_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        assert isinstance(actual, Sequence), path
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            assert_nested_equal(actual_item, expected_item, f"{path}[{index}]")
    else:
        assert actual == expected, path


def test_history_update_and_rollout_storage_keep_transition_order() -> None:
    torch.manual_seed(7)
    runner = make_runner()
    env = runner.env

    initial_obs = env.get_observations()
    runner._initialize_history(initial_obs)
    expected_initial = torch.zeros(NUM_ENVS, HISTORY_LENGTH, POLICY_DIM)
    expected_initial[:, -1] = initial_obs["policy"]
    assert torch.equal(runner.history, expected_initial)

    runner.learn(1)

    stored_history = runner.alg.storage.history.reshape(NUM_STEPS, NUM_ENVS, HISTORY_LENGTH, POLICY_DIM)
    obs0 = env.observations_at(0)["policy"]
    obs1 = env.observations_at(1)["policy"]
    obs2 = env.observations_at(2)["policy"]

    assert torch.equal(stored_history[0, :, -1], obs0)
    assert torch.equal(stored_history[1, :, -2:], torch.stack((obs0, obs1), dim=1))

    # Env 0 terminated after step 1, so only its new episode's obs2 remains.
    assert torch.count_nonzero(stored_history[2, 0, :-1]) == 0
    assert torch.equal(stored_history[2, 0, -1], obs2[0])
    assert torch.equal(stored_history[2, 1], torch.stack((obs0[1], obs1[1], obs2[1])))
    assert runner.alg.storage.step == 0


def test_learn_save_and_load_restore_model_and_both_optimizers(tmp_path) -> None:
    torch.manual_seed(11)
    runner = make_runner()
    ppo_before = [parameter.detach().clone() for parameter in runner.alg.ppo_parameters]
    student_parameters = list(runner.alg.policy.student_encoder.parameters())
    student_before = [parameter.detach().clone() for parameter in student_parameters]

    runner.learn(1)

    assert runner.env.step_count == NUM_STEPS
    assert runner.current_learning_iteration == 1
    assert any(not torch.equal(old, new) for old, new in zip(ppo_before, runner.alg.ppo_parameters))
    assert any(not torch.equal(old, new) for old, new in zip(student_before, student_parameters))
    assert runner.alg.optimizer.state
    assert runner.alg.student_optimizer.state

    checkpoint_path = tmp_path / "cts_runner.pt"
    runner.save(str(checkpoint_path), infos={"test": "cts"})
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    assert set(checkpoint) == {
        "model_state_dict",
        "optimizer_state_dict",
        "student_optimizer_state_dict",
        "iter",
        "infos",
    }
    assert checkpoint["optimizer_state_dict"]["state"]
    assert checkpoint["student_optimizer_state_dict"]["state"]
    assert checkpoint["iter"] == 1

    torch.manual_seed(1234)
    restored = make_runner()
    restored.history.fill_(99.0)
    restored.load(str(checkpoint_path), load_optimizer=True, map_location="cpu")

    assert_nested_equal(restored.alg.policy.state_dict(), runner.alg.policy.state_dict(), "model")
    assert_nested_equal(restored.alg.optimizer.state_dict(), runner.alg.optimizer.state_dict(), "optimizer")
    assert_nested_equal(
        restored.alg.student_optimizer.state_dict(),
        runner.alg.student_optimizer.state_dict(),
        "student_optimizer",
    )
    assert restored.current_learning_iteration == runner.current_learning_iteration
    assert torch.count_nonzero(restored.history) == 0

    restored.learn(1)
    assert restored.current_learning_iteration == 2
