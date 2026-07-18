#!/usr/bin/env python3
"""Load one IsaacLab articulation and print its runtime metadata.

This intentionally avoids creating a Gym environment, terrain, managers, or an
RL runner.  Change ``--config`` to inspect another ``ArticulationCfg``.

Example:

    ./isaaclab.sh -p scripts/inspect_robot.py --headless
"""

from __future__ import annotations

import argparse
import importlib

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect one IsaacLab ArticulationCfg.")
parser.add_argument(
    "--config",
    default="blank_rl_lab.assets.robot.unitree:UNITREE_GO2W_CFG",
    help="Robot config in 'python.module:VARIABLE_NAME' format.",
)
parser.add_argument(
    "--joint-pattern",
    action="append",
    default=[],
    help="Regex matched against runtime joint names. May be specified multiple times.",
)
parser.add_argument(
    "--body-pattern",
    action="append",
    default=[],
    help="Regex matched against runtime body names. May be specified multiple times.",
)
parser.add_argument(
    "--show-config",
    action="store_true",
    help="Also print the complete ArticulationCfg before spawning the robot.",
)
parser.add_argument(
    "--keep-alive",
    action="store_true",
    help="Keep stepping after printing, useful when running with the GUI.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402


def load_robot_cfg(spec: str) -> ArticulationCfg:
    """Import and copy an ArticulationCfg selected on the command line."""
    try:
        module_name, variable_name = spec.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError(f"Invalid --config {spec!r}; expected 'python.module:VARIABLE_NAME'.") from exc

    module = importlib.import_module(module_name)
    cfg = getattr(module, variable_name)
    if not isinstance(cfg, ArticulationCfg):
        raise TypeError(f"{spec} is {type(cfg).__name__}, not ArticulationCfg.")
    return cfg.copy()


def print_indexed(title: str, names: list[str]) -> None:
    print(f"\n{title} ({len(names)}):", flush=True)
    for index, name in enumerate(names):
        print(f"  [{index:02d}] {name}", flush=True)


def print_matches(robot: Articulation, kind: str, patterns: list[str]) -> None:
    finder = robot.find_joints if kind == "joint" else robot.find_bodies
    for pattern in patterns:
        indices, names = finder(pattern, preserve_order=True)
        print(f"\n{kind} pattern {pattern!r}:")
        if not names:
            print("  <no matches>")
        for index, name in zip(indices, names):
            print(f"  [{index:02d}] {name}")


def main() -> None:
    cfg = load_robot_cfg(args_cli.config)
    cfg.prim_path = "/World/Robot"

    if args_cli.show_config:
        print("\nArticulationCfg:")
        print(cfg)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    robot = Articulation(cfg)

    # reset() starts the simulation timeline and initializes the PhysX views.
    # Runtime joint/body names are only available after this point.
    print("[inspect] Initializing USD and PhysX views...", flush=True)
    sim.reset()
    print("[inspect] Initialization complete.", flush=True)

    print(f"\nConfig: {args_cli.config}")
    print(f"USD: {getattr(cfg.spawn, 'usd_path', '<not a USD spawn config>')}")
    print(f"Prim path: {cfg.prim_path}")
    print(f"Fixed base: {robot.is_fixed_base}")
    print_indexed("Body names", robot.body_names)
    print_indexed("Joint names", robot.joint_names)

    default_pos = robot.data.default_joint_pos[0].detach().cpu().tolist()
    default_vel = robot.data.default_joint_vel[0].detach().cpu().tolist()
    print("\nDefault joint state (runtime joint order):")
    for index, (name, position, velocity) in enumerate(zip(robot.joint_names, default_pos, default_vel)):
        print(f"  [{index:02d}] {name:<24} pos={position: .6f}  vel={velocity: .6f}")

    print_matches(robot, "body", args_cli.body_pattern)
    print_matches(robot, "joint", args_cli.joint_pattern)

    if args_cli.keep_alive:
        print("\nKeeping the simulator alive. Close the app or press Ctrl+C to stop.")
        while simulation_app.is_running():
            sim.step()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
