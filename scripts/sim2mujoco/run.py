#!/usr/bin/env python3
"""Run an exported IsaacLab policy inside MuJoCo.

This is the main file to read first. The loop is intentionally simple:

1. Read MuJoCo state in policy joint order.
2. Build the exact IsaacLab policy observation.
3. Evaluate the exported policy.
4. Convert action to joint position target.
5. Apply MuJoCo actuator controls.
6. Step physics and update the viewer.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from sim2mujoco.command import CommandState
from sim2mujoco.config import Sim2MujocoConfig
from sim2mujoco.control import JointPositionController
from sim2mujoco.mujoco_adapter import MujocoRobot
from sim2mujoco.observation import make_observation_builder
from sim2mujoco.policy import load_policy
from sim2mujoco.viewer import NullViewer, PassiveViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay an IsaacLab policy in MuJoCo.")
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/sim2mujoco/config/go2_amp_template.yaml",
        help="Path to the sim2mujoco YAML config.",
    )
    parser.add_argument("--model", type=str, default=None, help="Override robot.model_path.")
    parser.add_argument("--policy", type=str, default=None, help="Override policy.path.")
    parser.add_argument(
        "--command",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "WZ"),
        default=None,
        help="Override velocity command sent to the policy.",
    )
    parser.add_argument("--duration", type=float, default=None, help="Override run.duration_s. Use <=0 for no limit.")
    parser.add_argument("--headless", action="store_true", help="Run without viewer.")
    parser.add_argument("--zero-policy", action="store_true", help="Use zero actions instead of loading a policy.")
    parser.add_argument("--print-model-names", action="store_true", help="Print MuJoCo joint/actuator names and exit.")
    return parser.parse_args()

# 先读 YAML，再用命令行覆盖部分字段
def apply_cli_overrides(cfg: Sim2MujocoConfig, args: argparse.Namespace) -> Sim2MujocoConfig:
    if args.model is not None:
        cfg.robot.model_path = args.model
    if args.policy is not None:
        cfg.policy.path = args.policy
    if args.command is not None:
        cfg.command.lin_vel_x = float(args.command[0])
        cfg.command.lin_vel_y = float(args.command[1])
        cfg.command.ang_vel_z = float(args.command[2])
    if args.duration is not None:
        cfg.run.duration_s = float(args.duration)
    if args.headless:
        cfg.viewer.enabled = False
    if args.zero_policy:
        cfg.run.zero_policy = True
    if args.print_model_names:
        cfg.run.print_model_names = True
    return cfg


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(Sim2MujocoConfig.from_yaml(args.config), args)

    robot = MujocoRobot(cfg.robot, cfg.control)
    if cfg.run.print_model_names:
        robot.print_model_names()
        return

    robot.reset()

    command_state = CommandState.from_config(cfg.command)
    controller = JointPositionController(cfg.control, robot.default_joint_pos)
    obs_builder = make_observation_builder(cfg.observation, command_state, robot.default_joint_pos)
    policy = load_policy(cfg.policy, robot.num_policy_joints, zero_policy=cfg.run.zero_policy)
    policy.reset()

    viewer_cls = PassiveViewer if cfg.viewer.enabled else NullViewer
    viewer = viewer_cls(robot.model, robot.data, cfg.viewer, command_state) if cfg.viewer.enabled else viewer_cls() # type:ignore

    last_action = np.zeros(robot.num_policy_joints, dtype=np.float32)
    q_des = robot.default_joint_pos.copy()
    step_count = 0
    start_time = time.time()
    sim_dt = float(robot.model.opt.timestep)

    print("[sim2mujoco] starting rollout")
    print(f"  model:  {Path(cfg.robot.model_path)}")
    print(f"  policy: {'zero-policy' if cfg.run.zero_policy else Path(cfg.policy.path)}")
    print(f"  dt:     {sim_dt:.4f} s")
    print(f"  policy: every {cfg.control.policy_decimation} sim steps")
    command_state.print_command()

    with viewer as active_viewer:
        while active_viewer.is_running():
            elapsed = time.time() - start_time
            if cfg.run.duration_s > 0.0 and elapsed >= cfg.run.duration_s:
                break

            if active_viewer.consume_reset_request():
                robot.reset()
                policy.reset()
                last_action[:] = 0.0
                q_des = robot.default_joint_pos.copy()
                step_count = 0
                start_time = time.time()
                print("[sim2mujoco] reset")

            state = robot.read_state()
            if active_viewer.paused:
                active_viewer.sync(state.base_pos_w)
                active_viewer.sleep_when_paused()
                continue

            step_start = time.time()

            if step_count % cfg.control.policy_decimation == 0:
                obs = obs_builder.build(state, last_action)
                action = policy(obs)
                q_des = controller.action_to_position_target(action)
                last_action = np.asarray(action, dtype=np.float32)

            actuator_cmd = controller.compute_actuator_command(state.joint_pos, state.joint_vel, q_des)
            robot.apply_actuator_controls(actuator_cmd)
            robot.step()

            active_viewer.sync(state.base_pos_w)
            step_count += 1

            if cfg.viewer.realtime:
                sleep_time = sim_dt - (time.time() - step_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

    print("[sim2mujoco] finished")


if __name__ == "__main__":
    main()
