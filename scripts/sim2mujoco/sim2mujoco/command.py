"""Runtime velocity command state.

The YAML/CLI command is the initial value. In viewer mode, keyboard events can
update this state while the policy is running, and observation builders read the
latest value every policy step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CommandConfig


@dataclass
class CommandState:
    """Mutable velocity command: [lin_vel_x, lin_vel_y, ang_vel_z]."""

    lin_vel_x: float
    lin_vel_y: float
    ang_vel_z: float
    lin_vel_step: float
    ang_vel_step: float
    lin_vel_limit: float
    ang_vel_limit: float
    keyboard_enabled: bool = True

    @classmethod
    def from_config(cls, cfg: CommandConfig) -> "CommandState":
        return cls(
            lin_vel_x=cfg.lin_vel_x,
            lin_vel_y=cfg.lin_vel_y,
            ang_vel_z=cfg.ang_vel_z,
            lin_vel_step=cfg.lin_vel_step,
            ang_vel_step=cfg.ang_vel_step,
            lin_vel_limit=cfg.lin_vel_limit,
            ang_vel_limit=cfg.ang_vel_limit,
            keyboard_enabled=cfg.keyboard_enabled,
        )

    def as_array(self) -> np.ndarray:
        return np.asarray([self.lin_vel_x, self.lin_vel_y, self.ang_vel_z], dtype=np.float32)

    def reset(self) -> None:
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.ang_vel_z = 0.0
        self.print_command()

    def adjust(self, *, dx: float = 0.0, dy: float = 0.0, dyaw: float = 0.0) -> None:
        if not self.keyboard_enabled:
            return
        self.lin_vel_x = self._clip_linear(self.lin_vel_x + dx)
        self.lin_vel_y = self._clip_linear(self.lin_vel_y + dy)
        self.ang_vel_z = self._clip_yaw(self.ang_vel_z + dyaw)
        self.print_command()

    def print_command(self) -> None:
        print(
            "[command] "
            f"vx={self.lin_vel_x:+.2f} m/s, "
            f"vy={self.lin_vel_y:+.2f} m/s, "
            f"wz={self.ang_vel_z:+.2f} rad/s"
        )

    def _clip_linear(self, value: float) -> float:
        return float(np.clip(value, -self.lin_vel_limit, self.lin_vel_limit))

    def _clip_yaw(self, value: float) -> float:
        return float(np.clip(value, -self.ang_vel_limit, self.ang_vel_limit))
