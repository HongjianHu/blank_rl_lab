#!/usr/bin/env python3
"""Dump IsaacLab runtime joint/action metadata for building a MuJoCo mapping.

Run this with IsaacLab's Python launcher, for example:

  ./isaaclab.sh -p scripts/sim2mujoco/inspect_isaaclab_task.py \
      --task Go2-AMP-velocity-v0 \
      --output /tmp/go2_amp_runtime.yaml \
      --headless

The output is the source of truth for isaac_joint_names and default_joint_pos.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

try:
    from isaaclab.app import AppLauncher
except ModuleNotFoundError:
    AppLauncher = None


parser = argparse.ArgumentParser(description="Inspect IsaacLab runtime joint/action order.")
parser.add_argument("--task", type=str, required=True, help="Registered IsaacLab task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point.")
parser.add_argument("--output", type=str, default="/tmp/isaaclab_runtime_info.yaml", help="YAML output path.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs to create for inspection.")
if AppLauncher is not None:
    AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if AppLauncher is None:
    raise RuntimeError(
        "IsaacLab is not importable in this interpreter. Run this helper through IsaacLab, for example: "
        "./isaaclab.sh -p scripts/sim2mujoco/inspect_isaaclab_task.py --task Go2-AMP-velocity-v0 --headless"
    )

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import yaml

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_tasks.utils.hydra import hydra_task_config

import blank_rl_lab.tasks  # noqa: F401


def _to_yaml_value(value: Any):
    """Convert common IsaacLab runtime objects into YAML-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, slice):
        return f"slice({value.start}, {value.stop}, {value.step})"
    if hasattr(value, "detach"):
        array = value.detach().cpu().numpy()
        return array.item() if array.shape == () else array.tolist()
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if not hasattr(converted, "item") else converted.item()
    if isinstance(value, dict):
        return {str(k): _to_yaml_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_yaml_value(item) for item in value]
    return repr(value)


def _ids_to_yaml_value(ids):
    if isinstance(ids, slice):
        return _to_yaml_value(ids)
    if hasattr(ids, "detach"):
        return [int(item) for item in ids.detach().cpu().numpy().tolist()]
    return [int(item) for item in ids]


def _callable_name(func: Any) -> str:
    if func is None:
        return ""
    module = getattr(func, "__module__", "")
    name = getattr(func, "__name__", func.__class__.__name__)
    return f"{module}.{name}" if module else name


def _collect_observation_terms(env_unwrapped: Any) -> dict[str, Any]:
    manager = getattr(env_unwrapped, "observation_manager", None)
    if manager is None:
        return {}

    groups: dict[str, Any] = {}
    for group_name, term_names in manager._group_obs_term_names.items():  # type: ignore[attr-defined]
        term_dims = manager._group_obs_term_dim[group_name]  # type: ignore[attr-defined]
        term_cfgs = manager._group_obs_term_cfgs[group_name]  # type: ignore[attr-defined]
        group_payload = {
            "concatenate_terms": bool(manager._group_obs_concatenate[group_name]),  # type: ignore[attr-defined]
            "group_dim": _to_yaml_value(manager._group_obs_dim[group_name]),  # type: ignore[attr-defined]
            "terms": [],
        }
        for index, (term_name, term_dim, term_cfg) in enumerate(zip(term_names, term_dims, term_cfgs)):
            term_payload = {
                "index": index,
                "name": term_name,
                "shape": _to_yaml_value(term_dim),
                "func": _callable_name(getattr(term_cfg, "func", None)),
            }
            for attr_name in ("scale", "clip", "history_length", "flatten_history_dim", "params"):
                if hasattr(term_cfg, attr_name):
                    term_payload[attr_name] = _to_yaml_value(getattr(term_cfg, attr_name))
            group_payload["terms"].append(term_payload)
        groups[group_name] = group_payload
    return groups


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg) -> None:
    del agent_cfg
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.log_dir = os.path.dirname(args_cli.output) or "."
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        # IsaacLab's built-in descriptor exporter currently cannot serialize some
        # runtime tensors and non-decorated sensor terms. Collect the pieces we need
        # manually after environment construction instead.
        env_cfg.export_io_descriptors = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env) # type: ignore

    robot = env.unwrapped.scene["robot"] # type: ignore
    action_terms = {}
    if hasattr(env.unwrapped, "action_manager"):
        for name, term in env.unwrapped.action_manager._terms.items(): # type: ignore
            action_terms[name] = {
                "class": term.__class__.__name__,
                "joint_names": list(getattr(term, "_joint_names", [])),
                "joint_ids": _ids_to_yaml_value(getattr(term, "_joint_ids", [])),
                "action_dim": int(getattr(term, "action_dim", 0)),
            }
            for attr_name, output_name in (
                ("_scale", "scale"),
                ("_offset", "offset"),
                ("_clip", "clip"),
                ("_raw_actions", "raw_actions"),
            ):
                if hasattr(term, attr_name):
                    action_terms[name][output_name] = _to_yaml_value(getattr(term, attr_name))

    payload = {
        "task": args_cli.task,
        "env": {
            "decimation": int(getattr(env_cfg, "decimation", 0)),
            "sim_dt": float(getattr(env_cfg.sim, "dt", 0.0)) if hasattr(env_cfg, "sim") else 0.0,
            "step_dt": float(getattr(env.unwrapped, "step_dt", 0.0)),
        },
        "robot": {
            "joint_names": list(robot.joint_names),
            "body_names": list(robot.body_names),
            "default_joint_pos": robot.data.default_joint_pos[0].detach().cpu().numpy().tolist(),
            "default_joint_vel": robot.data.default_joint_vel[0].detach().cpu().numpy().tolist(),
        },
        "observations": _collect_observation_terms(env.unwrapped),
        "actions": action_terms,
        "notes": [
            "Use robot.joint_names as isaac_joint_names unless a specific action term uses a different _joint_names.",
            "Use actions.<term>.joint_names as the policy action order for that action term.",
            "Use observations.policy.terms as the concatenation order for the policy observation.",
        ],
    }

    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    print(f"[inspect] wrote {output}")
    env.close()


if __name__ == "__main__":
    main() # type: ignore
    simulation_app.close()
