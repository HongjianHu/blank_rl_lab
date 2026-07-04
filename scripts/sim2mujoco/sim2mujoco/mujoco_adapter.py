"""MuJoCo boundary layer.

This module is the only place that should know MuJoCo qpos/qvel/actuator
indices. The rest of the runner consumes policy-ordered vectors, which makes it
much easier to compare against IsaacLab.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ControlConfig, RobotConfig
from .math_utils import projected_gravity_from_quat, rotate_world_to_body


def _load_mujoco() -> Any:
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The Python package 'mujoco' is not installed in this interpreter. "
            "Install it in the environment you use to run sim2mujoco."
        ) from exc
    return mujoco


@dataclass
class RobotState:
    """Policy-ordered robot state read from MuJoCo."""

    base_pos_w: np.ndarray
    base_quat_wxyz: np.ndarray
    base_lin_vel_b: np.ndarray
    base_ang_vel_b: np.ndarray
    projected_gravity_b: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    joint_effort: np.ndarray


class MujocoRobot:
    """Thin wrapper around a MuJoCo model/data pair."""

    def __init__(self, robot_cfg: RobotConfig, control_cfg: ControlConfig):
        self.mujoco = _load_mujoco()
        self.cfg = robot_cfg

        model_path = Path(robot_cfg.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"MuJoCo model file does not exist: {model_path}")

        self.model = self.mujoco.MjModel.from_xml_path(str(model_path))
        self.data = self.mujoco.MjData(self.model)

        if control_cfg.sim_dt > 0.0:
            self.model.opt.timestep = float(control_cfg.sim_dt)

        self.isaac_joint_names = list(robot_cfg.isaac_joint_names or [])
        self.mujoco_joint_names = list(robot_cfg.mujoco_joint_names or [])
        self.default_joint_pos = np.asarray(robot_cfg.default_joint_pos, dtype=np.float64)

        self.joint_ids = [self._joint_id(name) for name in self.mujoco_joint_names]
        self.qpos_addrs = np.asarray([self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids], dtype=np.int32)
        self.qvel_addrs = np.asarray([self.model.jnt_dofadr[joint_id] for joint_id in self.joint_ids], dtype=np.int32)
        self.actuator_ids = self._resolve_actuators(robot_cfg)

    @property
    def num_policy_joints(self) -> int:
        return len(self.isaac_joint_names)

    def reset(self) -> None:
        """Reset MuJoCo and write the IsaacLab default pose in policy order."""
        self.mujoco.mj_resetData(self.model, self.data)

        if self.cfg.free_joint:
            self.data.qpos[:7] = np.asarray(self.cfg.base_qpos, dtype=np.float64)
            self.data.qvel[:6] = np.asarray(self.cfg.base_qvel, dtype=np.float64)

        self.data.qpos[self.qpos_addrs] = self.default_joint_pos
        self.data.qvel[self.qvel_addrs] = 0.0
        self.mujoco.mj_forward(self.model, self.data)

    def read_state(self) -> RobotState:
        """Read MuJoCo state and return policy-ordered arrays."""
        joint_pos = np.asarray(self.data.qpos[self.qpos_addrs], dtype=np.float64).copy()
        joint_vel = np.asarray(self.data.qvel[self.qvel_addrs], dtype=np.float64).copy()
        joint_effort = np.asarray(self.data.qfrc_actuator[self.qvel_addrs], dtype=np.float64).copy()

        if self.cfg.free_joint:
            base_pos_w = np.asarray(self.data.qpos[:3], dtype=np.float64).copy()
            base_quat_wxyz = np.asarray(self.data.qpos[3:7], dtype=np.float64).copy()
            base_lin_vel_w = np.asarray(self.data.qvel[:3], dtype=np.float64).copy()
            base_ang_vel_raw = np.asarray(self.data.qvel[3:6], dtype=np.float64).copy()
            base_lin_vel_b = rotate_world_to_body(base_quat_wxyz, base_lin_vel_w)
            if self.cfg.free_joint_ang_vel_frame == "world":
                base_ang_vel_b = rotate_world_to_body(base_quat_wxyz, base_ang_vel_raw)
            else:
                base_ang_vel_b = base_ang_vel_raw
            projected_gravity_b = projected_gravity_from_quat(base_quat_wxyz)
        else:
            base_pos_w = np.zeros(3, dtype=np.float64)
            base_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            base_lin_vel_b = np.zeros(3, dtype=np.float64)
            base_ang_vel_b = np.zeros(3, dtype=np.float64)
            projected_gravity_b = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        return RobotState(
            base_pos_w=base_pos_w,
            base_quat_wxyz=base_quat_wxyz,
            base_lin_vel_b=base_lin_vel_b,
            base_ang_vel_b=base_ang_vel_b,
            projected_gravity_b=projected_gravity_b,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            joint_effort=joint_effort,
        )

    def apply_actuator_controls(self, values: np.ndarray) -> None:
        """Write policy-ordered actuator controls to MuJoCo data.ctrl."""
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.num_policy_joints,):
            raise ValueError(f"Expected control shape {(self.num_policy_joints,)}, got {values.shape}.")
        self.data.ctrl[self.actuator_ids] = values

    def step(self) -> None:
        self.mujoco.mj_step(self.model, self.data)

    def print_model_names(self) -> None:
        """Print MuJoCo joint/actuator names for building the mapping YAML."""
        print("\n[MuJoCo] joints:")
        for joint_id in range(self.model.njnt):
            name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            print(f"  {joint_id:02d}: {name}")
        print("\n[MuJoCo] actuators:")
        for actuator_id in range(self.model.nu):
            name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            trn_joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            joint_name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, trn_joint_id)
            print(f"  {actuator_id:02d}: {name} -> joint {trn_joint_id}: {joint_name}")

    def _joint_id(self, joint_name: str) -> int:
        joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo joint not found: {joint_name}")
        if int(self.model.jnt_qposadr[joint_id]) < 0 or int(self.model.jnt_dofadr[joint_id]) < 0:
            raise ValueError(f"MuJoCo joint has invalid qpos/qvel address: {joint_name}")
        return int(joint_id)

    def _actuator_id(self, actuator_name: str) -> int:
        actuator_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise ValueError(f"MuJoCo actuator not found: {actuator_name}")
        return int(actuator_id)

    def _resolve_actuators(self, robot_cfg: RobotConfig) -> np.ndarray:
        if robot_cfg.mujoco_actuator_names:
            actuator_ids = [self._actuator_id(name) for name in robot_cfg.mujoco_actuator_names]
        else:
            actuator_ids = [self._find_actuator_attached_to_joint(joint_id) for joint_id in self.joint_ids]
        return np.asarray(actuator_ids, dtype=np.int32)

    def _find_actuator_attached_to_joint(self, joint_id: int) -> int:
        matches: list[int] = []
        for actuator_id in range(self.model.nu):
            trn_joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if trn_joint_id == joint_id:
                matches.append(actuator_id)
        if not matches:
            joint_name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            raise ValueError(
                f"No actuator found for MuJoCo joint '{joint_name}'. "
                "Set robot.mujoco_actuator_names explicitly in the YAML."
            )
        if len(matches) > 1:
            joint_name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            print(f"[WARN] Multiple actuators found for joint '{joint_name}', using actuator id {matches[0]}.")
        return matches[0]
