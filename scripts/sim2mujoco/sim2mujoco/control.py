"""Action post-processing and low-level joint control."""

from __future__ import annotations

import numpy as np

from .config import ControlConfig


class JointPositionController:
    """Convert policy actions into MuJoCo actuator controls.

    IsaacLab JointPositionAction does:

        processed_action = default_joint_pos + action_scale * raw_action

    If the MuJoCo model uses torque motors, this class then applies a joint PD
    law around that processed joint-position target. If the model uses position
    actuators, it can instead return the position target directly.
    """

    def __init__(self, cfg: ControlConfig, default_joint_pos: np.ndarray):
        self.cfg = cfg
        self.default_joint_pos = np.asarray(default_joint_pos, dtype=np.float64)
        self.action_scale = self._expand_action_scale(cfg.action_scale, self.default_joint_pos.shape[0])

    def action_to_position_target(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.default_joint_pos.shape:
            raise ValueError(f"Expected action shape {self.default_joint_pos.shape}, got {action.shape}.")
        if self.cfg.action_clip is not None:
            action = np.clip(action, self.cfg.action_clip[0], self.cfg.action_clip[1])
        return self.default_joint_pos + self.action_scale * action

    def compute_actuator_command(self, q: np.ndarray, qd: np.ndarray, q_des: np.ndarray) -> np.ndarray:
        """Return torque or position commands in policy joint order."""
        if self.cfg.actuator_mode == "position":
            return q_des

        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        q_des = np.asarray(q_des, dtype=np.float64)
        torque = self.cfg.kp * (q_des - q) - self.cfg.kd * qd
        return np.clip(torque, -self.cfg.torque_limit, self.cfg.torque_limit)

    @staticmethod
    def _expand_action_scale(action_scale: float | list[float], num_joints: int) -> np.ndarray:
        if isinstance(action_scale, list):
            return np.asarray(action_scale, dtype=np.float64)
        return np.full(num_joints, float(action_scale), dtype=np.float64)
