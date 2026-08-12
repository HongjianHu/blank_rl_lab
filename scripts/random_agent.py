# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many policy steps. A negative value runs until Isaac Sim closes.")
parser.add_argument("--action_amplitude", type=float, default=0.2, help="Amplitude of uniform normalized random actions in [0, 1].")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import blank_rl_lab.tasks  # noqa: F401

def _iter_named_tensors(
    value,
    prefix: str,
):
    """Recursively iterate over tensors in dict/TensorDict data."""

    if isinstance(value, torch.Tensor):
        yield prefix, value
        return

    if hasattr(value, "items"):
        for key, child_value in value.items():
            child_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )
            yield from _iter_named_tensors(
                child_value,
                child_prefix,
            )


def _assert_environment_finite(env, observations, rewards: torch.Tensor, step_index: int,) -> None:
    """Fail immediately when the environment produces NaN or Inf."""

    unwrapped_env = env.unwrapped
    robot = unwrapped_env.scene["robot"]

    named_tensors = list(
        _iter_named_tensors(
            observations,
            "observations",
        )
    )

    named_tensors.extend(
        [
            ("rewards", rewards),
            (
                "root_pos_w",
                robot.data.root_pos_w,
            ),
            (
                "root_quat_w",
                robot.data.root_quat_w,
            ),
            (
                "joint_pos",
                robot.data.joint_pos,
            ),
            (
                "joint_vel",
                robot.data.joint_vel,
            ),
            (
                "actions",
                unwrapped_env.action_manager.action,
            ),
        ]
    )

    if "waypoint" in unwrapped_env.command_manager.active_terms:
        named_tensors.append(
            (
                "waypoint_command",
                unwrapped_env.command_manager.get_command("waypoint"),
            )
        )

    non_finite_names = []

    for name, tensor in named_tensors:
        if not tensor.is_floating_point():
            continue

        if not torch.isfinite(tensor).all():
            non_finite_names.append(name)

    if non_finite_names:
        raise RuntimeError(
            f"Non-finite tensors at step {step_index}: "
            f"{non_finite_names}"
        )

def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    observations, _ = env.reset()

    if args_cli.max_steps < -1:
        raise ValueError("--max_steps must be -1 or non-negative.")

    if not 0.0 <= args_cli.action_amplitude <= 1.0:
        raise ValueError("--action_amplitude must be in [0, 1].")

    step_index = 0
    # simulate environment
    while simulation_app.is_running() and (args_cli.max_steps < 0 or step_index < args_cli.max_steps):
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = 2 * torch.rand_like(env.unwrapped.action_manager.action) - 1 # type:ignore
            actions *= args_cli.action_amplitude

            (observations, rewards, terminated, truncated, extras,) = env.step(actions)

            _assert_environment_finite(env, observations, rewards, step_index,) # type:ignore

            step_index += 1
    if args_cli.max_steps >= 0:
        print(
            "[PASS] Random-action smoke test completed:",
            f"{step_index} policy steps,",
            f"{env.unwrapped.num_envs} environments,", # type:ignore
            "no NaN or Inf.",
        )
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
