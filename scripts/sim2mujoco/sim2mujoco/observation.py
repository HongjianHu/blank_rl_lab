"""Observation builders.

Each builder must reproduce the exact policy observation used during IsaacLab
training. Start with one task and keep it boringly explicit; it is much easier
to debug one 45-dimensional vector than a generic abstraction.
"""

from __future__ import annotations

import numpy as np

from .command import CommandState
from .config import ObservationConfig
from .mujoco_adapter import RobotState


class Go2AmpObservationBuilder:
    """Build the policy observation from amp_go2_velocity.py.

    Policy order:
      base_ang_vel * 0.25
      projected_gravity
      velocity_commands
      joint_pos_rel
      joint_vel_rel * 0.05
      last_action
    """

    def __init__(self, cfg: ObservationConfig, command_state: CommandState, default_joint_pos: np.ndarray):
        self.cfg = cfg
        self.command_state = command_state
        self.default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32)

    def build(self, state: RobotState, last_action: np.ndarray) -> np.ndarray:
        joint_pos_rel = (state.joint_pos.astype(np.float32) - self.default_joint_pos) * self.cfg.joint_pos_scale
        joint_vel_rel = state.joint_vel.astype(np.float32) * self.cfg.joint_vel_scale
        obs = np.concatenate(
            [
                state.base_ang_vel_b.astype(np.float32) * self.cfg.base_ang_vel_scale,
                state.projected_gravity_b.astype(np.float32),
                self.command_state.as_array(),
                joint_pos_rel,
                joint_vel_rel,
                np.asarray(last_action, dtype=np.float32),
            ],
            dtype=np.float32,
        )
        if obs.shape[0] != self.cfg.expected_dim:
            raise ValueError(f"Expected obs dim {self.cfg.expected_dim}, got {obs.shape[0]}.")
        return obs


class Go2RoughVelocityObservationBuilder:
    """Build the policy observation from Go2-rough-velocity-v0.

    Policy order exported by inspect_isaaclab_task.py:
      base_lin_vel * 1.0
      base_ang_vel * 0.2
      projected_gravity
      velocity_commands
      joint_pos_rel
      joint_vel_rel * 0.05
      joint_effort * 0.01
      last_action
      height_scanner

    The first implementation supports a flat-ground height scan. This keeps the
    vector contract correct while the rest of the sim-to-sim stack is validated.
    Replace it with MuJoCo ray casts before judging rough-terrain performance.
    """

    def __init__(self, cfg: ObservationConfig, command_state: CommandState, default_joint_pos: np.ndarray):
        self.cfg = cfg
        self.command_state = command_state
        self.default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32)

    def build(self, state: RobotState, last_action: np.ndarray) -> np.ndarray:
        joint_pos_rel = (state.joint_pos.astype(np.float32) - self.default_joint_pos) * self.cfg.joint_pos_scale
        joint_vel_rel = state.joint_vel.astype(np.float32) * self.cfg.joint_vel_scale
        joint_effort = state.joint_effort.astype(np.float32) * self.cfg.joint_effort_scale
        height_scan = self._height_scan(state)
        obs = np.concatenate(
            [
                state.base_lin_vel_b.astype(np.float32) * self.cfg.base_lin_vel_scale,
                state.base_ang_vel_b.astype(np.float32) * self.cfg.base_ang_vel_scale,
                state.projected_gravity_b.astype(np.float32),
                self.command_state.as_array(),
                joint_pos_rel,
                joint_vel_rel,
                joint_effort,
                np.asarray(last_action, dtype=np.float32),
                height_scan,
            ],
            dtype=np.float32,
        )
        if obs.shape[0] != self.cfg.expected_dim:
            raise ValueError(f"Expected obs dim {self.cfg.expected_dim}, got {obs.shape[0]}.")
        return obs

    def _height_scan(self, state: RobotState) -> np.ndarray:
        if self.cfg.height_scan_mode != "flat":
            raise NotImplementedError(
                f"Unsupported height_scan_mode '{self.cfg.height_scan_mode}'. "
                "Only 'flat' is implemented in this first rough-policy scaffold."
            )
        value = float(state.base_pos_w[2]) - self.cfg.height_scan_ground_z - self.cfg.height_scan_offset
        height_scan = np.full(self.cfg.height_scan_num_rays, value, dtype=np.float32)
        if self.cfg.height_scan_clip is not None:
            height_scan = np.clip(height_scan, self.cfg.height_scan_clip[0], self.cfg.height_scan_clip[1])
        return height_scan


class Go2TsVelocityObservationBuilder:
    """Build the 57-D distilled student observation from Go2-ts-velocity-v0.

    Policy order exported by /tmp/go2_ts_runtime.yaml:
      base_ang_vel * 0.2
      projected_gravity
      velocity_commands
      joint_pos_rel
      joint_vel_rel * 0.05
      joint_effort * 0.01
      last_action

    The teacher/critic group is 247-D and includes privileged information; it is
    intentionally not used for deployment.
    """

    def __init__(self, cfg: ObservationConfig, command_state: CommandState, default_joint_pos: np.ndarray):
        self.cfg = cfg
        self.command_state = command_state
        self.default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32)

    def build(self, state: RobotState, last_action: np.ndarray) -> np.ndarray:
        joint_pos_rel = (state.joint_pos.astype(np.float32) - self.default_joint_pos) * self.cfg.joint_pos_scale
        joint_vel_rel = state.joint_vel.astype(np.float32) * self.cfg.joint_vel_scale
        joint_effort = state.joint_effort.astype(np.float32) * self.cfg.joint_effort_scale
        obs = np.concatenate(
            [
                state.base_ang_vel_b.astype(np.float32) * self.cfg.base_ang_vel_scale,
                state.projected_gravity_b.astype(np.float32),
                self.command_state.as_array(),
                joint_pos_rel,
                joint_vel_rel,
                joint_effort,
                np.asarray(last_action, dtype=np.float32),
            ],
            dtype=np.float32,
        )
        if obs.shape[0] != self.cfg.expected_dim:
            raise ValueError(f"Expected obs dim {self.cfg.expected_dim}, got {obs.shape[0]}.")
        return obs


def make_observation_builder(
    cfg: ObservationConfig, command_state: CommandState, default_joint_pos: np.ndarray
) -> Go2AmpObservationBuilder | Go2RoughVelocityObservationBuilder | Go2TsVelocityObservationBuilder:
    """Factory for observation builders.

    Add a new builder here when you migrate another IsaacLab task. For example,
    GO2RobotDemoEnvCfg needs base_lin_vel, joint_effort, and height_scan, which
    are intentionally not faked inside the AMP builder.
    """
    if cfg.kind == "go2_amp":
        return Go2AmpObservationBuilder(cfg, command_state, default_joint_pos)
    if cfg.kind == "go2_rough_velocity":
        return Go2RoughVelocityObservationBuilder(cfg, command_state, default_joint_pos)
    if cfg.kind == "go2_ts_velocity":
        return Go2TsVelocityObservationBuilder(cfg, command_state, default_joint_pos)
    raise NotImplementedError(f"Unsupported observation.kind: {cfg.kind}")
