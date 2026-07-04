"""Configuration loader for the sim-to-MuJoCo runner.

The runner is deliberately driven by a YAML file. Sim-to-sim failures are often
caused by hidden assumptions about joint order, action scale, and default pose,
so those values are kept visible instead of being buried in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{name}' must be a mapping.")
    return value


def _float_list(value: Any, name: str) -> list[float]:
    if not isinstance(value, list):
        raise TypeError(f"'{name}' must be a list of numbers.")
    return [float(item) for item in value]


def _str_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"'{name}' must be a list of strings.")
    return [str(item) for item in value]


@dataclass
class PolicyConfig:
    path: str = ""
    backend: str = "torchscript"
    device: str = "cpu"


@dataclass
class RobotConfig:
    model_path: str = ""
    free_joint: bool = True
    free_joint_ang_vel_frame: str = "body"
    base_qpos: list[float] | None = None
    base_qvel: list[float] | None = None
    isaac_joint_names: list[str] | None = None
    mujoco_joint_names: list[str] | None = None
    mujoco_actuator_names: list[str] | None = None
    default_joint_pos: list[float] | None = None

    def validate(self) -> None:
        self.base_qpos = self.base_qpos or [0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
        self.base_qvel = self.base_qvel or [0.0] * 6
        self.isaac_joint_names = self.isaac_joint_names or []
        self.mujoco_joint_names = self.mujoco_joint_names or list(self.isaac_joint_names)
        self.mujoco_actuator_names = self.mujoco_actuator_names or []
        self.default_joint_pos = self.default_joint_pos or [0.0] * len(self.isaac_joint_names)

        n = len(self.isaac_joint_names)
        if n == 0:
            raise ValueError("robot.isaac_joint_names must not be empty.")
        if len(self.mujoco_joint_names) != n:
            raise ValueError("robot.mujoco_joint_names must have the same length as robot.isaac_joint_names.")
        if self.mujoco_actuator_names and len(self.mujoco_actuator_names) != n:
            raise ValueError("robot.mujoco_actuator_names must be empty or match robot.isaac_joint_names length.")
        if len(self.default_joint_pos) != n:
            raise ValueError("robot.default_joint_pos must match robot.isaac_joint_names length.")
        if self.free_joint and len(self.base_qpos) != 7:
            raise ValueError("robot.base_qpos must contain 7 values for a free joint: x y z qw qx qy qz.")
        if self.free_joint and len(self.base_qvel) != 6:
            raise ValueError("robot.base_qvel must contain 6 values for a free joint.")
        if self.free_joint_ang_vel_frame not in {"body", "world"}:
            raise ValueError("robot.free_joint_ang_vel_frame must be 'body' or 'world'.")


@dataclass
class ControlConfig:
    sim_dt: float = 0.005
    policy_decimation: int = 4
    kp: float = 25.0
    kd: float = 0.5
    torque_limit: float = 23.5
    action_scale: float | list[float] = 0.25
    action_clip: list[float] | None = None
    actuator_mode: str = "torque"

    def validate(self, num_joints: int) -> None:
        if self.policy_decimation < 1:
            raise ValueError("control.policy_decimation must be >= 1.")
        if isinstance(self.action_scale, list) and len(self.action_scale) != num_joints:
            raise ValueError("control.action_scale must be a scalar or one value per joint.")
        if self.action_clip is not None and len(self.action_clip) != 2:
            raise ValueError("control.action_clip must be [min, max].")
        if self.actuator_mode not in {"torque", "position"}:
            raise ValueError("control.actuator_mode must be 'torque' or 'position'.")


@dataclass
class ObservationConfig:
    kind: str = "go2_amp"
    expected_dim: int = 45
    base_lin_vel_scale: float = 1.0
    base_ang_vel_scale: float = 0.25
    joint_pos_scale: float = 1.0
    joint_vel_scale: float = 0.05
    joint_effort_scale: float = 0.01
    height_scan_mode: str = "flat"
    height_scan_num_rays: int = 187
    height_scan_offset: float = 0.5
    height_scan_ground_z: float = 0.0
    height_scan_clip: list[float] | None = None


@dataclass
class CommandConfig:
    lin_vel_x: float = 0.0
    lin_vel_y: float = 0.0
    ang_vel_z: float = 0.0
    keyboard_enabled: bool = True
    lin_vel_step: float = 0.1
    ang_vel_step: float = 0.1
    lin_vel_limit: float = 1.5
    ang_vel_limit: float = 1.0

    def as_array(self) -> list[float]:
        return [self.lin_vel_x, self.lin_vel_y, self.ang_vel_z]


@dataclass
class ViewerConfig:
    enabled: bool = True
    realtime: bool = True
    follow_base: bool = True
    distance: float = 2.5
    azimuth: float = 135.0
    elevation: float = -20.0
    lookat: list[float] | None = None


@dataclass
class RunConfig:
    duration_s: float = 20.0
    zero_policy: bool = False
    print_model_names: bool = False


@dataclass
class Sim2MujocoConfig:
    policy: PolicyConfig
    robot: RobotConfig
    control: ControlConfig
    observation: ObservationConfig
    command: CommandConfig
    viewer: ViewerConfig
    run: RunConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Sim2MujocoConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise TypeError("Top-level YAML config must be a mapping.")

        policy_section = _section(data, "policy")
        robot_section = _section(data, "robot")
        control_section = _section(data, "control")
        observation_section = _section(data, "observation")
        command_section = _section(data, "command")
        viewer_section = _section(data, "viewer")
        run_section = _section(data, "run")

        robot = RobotConfig(
            model_path=str(robot_section.get("model_path", "")),
            free_joint=bool(robot_section.get("free_joint", True)),
            free_joint_ang_vel_frame=str(robot_section.get("free_joint_ang_vel_frame", "body")),
            base_qpos=_float_list(robot_section.get("base_qpos", [0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]), "base_qpos"),
            base_qvel=_float_list(robot_section.get("base_qvel", [0.0] * 6), "base_qvel"),
            isaac_joint_names=_str_list(robot_section.get("isaac_joint_names", []), "isaac_joint_names"),
            mujoco_joint_names=_str_list(
                robot_section.get("mujoco_joint_names", robot_section.get("isaac_joint_names", [])),
                "mujoco_joint_names",
            ),
            mujoco_actuator_names=_str_list(robot_section.get("mujoco_actuator_names", []), "mujoco_actuator_names"),
            default_joint_pos=_float_list(robot_section.get("default_joint_pos", []), "default_joint_pos"),
        )
        robot.validate()

        action_scale: float | list[float]
        raw_action_scale = control_section.get("action_scale", 0.25)
        if isinstance(raw_action_scale, list):
            action_scale = [float(item) for item in raw_action_scale]
        else:
            action_scale = float(raw_action_scale)

        control = ControlConfig(
            sim_dt=float(control_section.get("sim_dt", 0.005)),
            policy_decimation=int(control_section.get("policy_decimation", 4)),
            kp=float(control_section.get("kp", 25.0)),
            kd=float(control_section.get("kd", 0.5)),
            torque_limit=float(control_section.get("torque_limit", 23.5)),
            action_scale=action_scale,
            action_clip=_float_list(control_section["action_clip"], "action_clip")
            if "action_clip" in control_section
            else None,
            actuator_mode=str(control_section.get("actuator_mode", "torque")),
        )
        control.validate(len(robot.isaac_joint_names or []))

        cfg = cls(
            policy=PolicyConfig(
                path=str(policy_section.get("path", "")),
                backend=str(policy_section.get("backend", "torchscript")),
                device=str(policy_section.get("device", "cpu")),
            ),
            robot=robot,
            control=control,
            observation=ObservationConfig(
                kind=str(observation_section.get("kind", "go2_amp")),
                expected_dim=int(observation_section.get("expected_dim", 45)),
                base_lin_vel_scale=float(observation_section.get("base_lin_vel_scale", 1.0)),
                base_ang_vel_scale=float(observation_section.get("base_ang_vel_scale", 0.25)),
                joint_pos_scale=float(observation_section.get("joint_pos_scale", 1.0)),
                joint_vel_scale=float(observation_section.get("joint_vel_scale", 0.05)),
                joint_effort_scale=float(observation_section.get("joint_effort_scale", 0.01)),
                height_scan_mode=str(observation_section.get("height_scan_mode", "flat")),
                height_scan_num_rays=int(observation_section.get("height_scan_num_rays", 187)),
                height_scan_offset=float(observation_section.get("height_scan_offset", 0.5)),
                height_scan_ground_z=float(observation_section.get("height_scan_ground_z", 0.0)),
                height_scan_clip=_float_list(observation_section["height_scan_clip"], "height_scan_clip")
                if "height_scan_clip" in observation_section
                else None,
            ),
            command=CommandConfig(
                lin_vel_x=float(command_section.get("lin_vel_x", 0.0)),
                lin_vel_y=float(command_section.get("lin_vel_y", 0.0)),
                ang_vel_z=float(command_section.get("ang_vel_z", 0.0)),
                keyboard_enabled=bool(command_section.get("keyboard_enabled", True)),
                lin_vel_step=float(command_section.get("lin_vel_step", 0.1)),
                ang_vel_step=float(command_section.get("ang_vel_step", 0.1)),
                lin_vel_limit=float(command_section.get("lin_vel_limit", 1.5)),
                ang_vel_limit=float(command_section.get("ang_vel_limit", 1.0)),
            ),
            viewer=ViewerConfig(
                enabled=bool(viewer_section.get("enabled", True)),
                realtime=bool(viewer_section.get("realtime", True)),
                follow_base=bool(viewer_section.get("follow_base", True)),
                distance=float(viewer_section.get("distance", 2.5)),
                azimuth=float(viewer_section.get("azimuth", 135.0)),
                elevation=float(viewer_section.get("elevation", -20.0)),
                lookat=_float_list(viewer_section.get("lookat", [0.0, 0.0, 0.35]), "lookat"),
            ),
            run=RunConfig(
                duration_s=float(run_section.get("duration_s", 20.0)),
                zero_policy=bool(run_section.get("zero_policy", False)),
                print_model_names=bool(run_section.get("print_model_names", False)),
            ),
        )
        return cfg
