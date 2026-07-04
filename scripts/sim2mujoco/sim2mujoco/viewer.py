"""MuJoCo passive viewer helpers with simple camera controls."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .command import CommandState
from .config import ViewerConfig


GLFW_KEY_RIGHT = 262
GLFW_KEY_LEFT = 263
GLFW_KEY_DOWN = 264
GLFW_KEY_UP = 265


@dataclass
class CameraController:
    """Keyboard state and camera manipulation for the passive viewer."""

    cfg: ViewerConfig
    command_state: CommandState | None = None
    paused: bool = False
    reset_requested: bool = False

    def print_help(self) -> None:
        print(
            "\n[viewer keys]\n"
            "  h: print this help\n"
            "  p: pause/resume physics\n"
            "  r: reset simulation\n"
            "  f: toggle follow-base camera\n"
            "  w/s: raise/lower camera elevation\n"
            "  a/d: rotate camera azimuth\n"
            "  q/e: zoom in/out\n"
            "  i/k: move lookat forward/backward in world x\n"
            "  j/l: move lookat left/right in world y\n"
            "  u/o: move lookat down/up in world z\n"
            "\n[command keys]\n"
            "  up/down: increase/decrease vx\n"
            "  left/right: increase/decrease yaw rate wz\n"
            "  z/x: increase/decrease lateral velocity vy\n"
            "  space: set vx, vy, wz to zero\n"
        )

    def on_key(self, keycode: int, cam: Any) -> None:
        """Handle one key event from mujoco.viewer.launch_passive."""
        key = chr(keycode).lower() if 0 <= keycode < 256 else ""
        if self._handle_command_key(keycode, key):
            return
        if key == "h":
            self.print_help()
        elif key == "p":
            self.paused = not self.paused
            print(f"[viewer] paused={self.paused}")
        elif key == "r":
            self.reset_requested = True
        elif key == "f":
            self.cfg.follow_base = not self.cfg.follow_base
            print(f"[viewer] follow_base={self.cfg.follow_base}")
        elif key == "w":
            cam.elevation += 5.0
        elif key == "s":
            cam.elevation -= 5.0
        elif key == "a":
            cam.azimuth -= 5.0
        elif key == "d":
            cam.azimuth += 5.0
        elif key == "q":
            cam.distance = max(0.1, cam.distance * 0.9)
        elif key == "e":
            cam.distance = cam.distance * 1.1
        elif key == "i":
            cam.lookat[0] += 0.05
        elif key == "k":
            cam.lookat[0] -= 0.05
        elif key == "j":
            cam.lookat[1] += 0.05
        elif key == "l":
            cam.lookat[1] -= 0.05
        elif key == "u":
            cam.lookat[2] -= 0.05
        elif key == "o":
            cam.lookat[2] += 0.05

    def _handle_command_key(self, keycode: int, key: str) -> bool:
        if self.command_state is None:
            return False
        if keycode == GLFW_KEY_UP:
            self.command_state.adjust(dx=self.command_state.lin_vel_step)
        elif keycode == GLFW_KEY_DOWN:
            self.command_state.adjust(dx=-self.command_state.lin_vel_step)
        elif keycode == GLFW_KEY_LEFT:
            self.command_state.adjust(dyaw=self.command_state.ang_vel_step)
        elif keycode == GLFW_KEY_RIGHT:
            self.command_state.adjust(dyaw=-self.command_state.ang_vel_step)
        elif key == "z":
            self.command_state.adjust(dy=self.command_state.lin_vel_step)
        elif key == "x":
            self.command_state.adjust(dy=-self.command_state.lin_vel_step)
        elif key == " ":
            self.command_state.reset()
        else:
            return False
        return True

    def consume_reset_request(self) -> bool:
        requested = self.reset_requested
        self.reset_requested = False
        return requested


class PassiveViewer:
    """Context manager around mujoco.viewer.launch_passive."""

    def __init__(self, model: Any, data: Any, cfg: ViewerConfig, command_state: CommandState | None = None):
        self.model = model
        self.data = data
        self.cfg = cfg
        self.controller = CameraController(cfg, command_state)
        self._viewer_module: Any | None = None
        self._context: Any | None = None
        self._viewer: Any | None = None

    @property
    def paused(self) -> bool:
        return self.controller.paused

    def __enter__(self) -> "PassiveViewer":
        self._viewer_module = importlib.import_module("mujoco.viewer")

        def key_callback(keycode: int) -> None:
            if self._viewer is not None:
                self.controller.on_key(keycode, self._viewer.cam)

        self._context = self._viewer_module.launch_passive(self.model, self.data, key_callback=key_callback)
        self._viewer = self._context.__enter__() # type: ignore
        self._configure_camera()
        self.controller.print_help()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._context is not None:
            self._context.__exit__(exc_type, exc, tb)

    def is_running(self) -> bool:
        return bool(self._viewer is not None and self._viewer.is_running())

    def consume_reset_request(self) -> bool:
        return self.controller.consume_reset_request()

    def sync(self, base_pos_w: np.ndarray | None = None) -> None:
        if self._viewer is None:
            return
        if self.cfg.follow_base and base_pos_w is not None:
            self._viewer.cam.lookat[:] = np.asarray(base_pos_w, dtype=np.float64)
        self._viewer.sync()

    def sleep_when_paused(self) -> None:
        time.sleep(0.02)

    def _configure_camera(self) -> None:
        if self._viewer is None:
            return
        self._viewer.cam.distance = self.cfg.distance
        self._viewer.cam.azimuth = self.cfg.azimuth
        self._viewer.cam.elevation = self.cfg.elevation
        self._viewer.cam.lookat[:] = np.asarray(self.cfg.lookat or [0.0, 0.0, 0.35], dtype=np.float64)


class NullViewer:
    """Headless viewer replacement used for automated checks."""

    paused = False

    def __enter__(self) -> "NullViewer":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def is_running(self) -> bool:
        return True

    def consume_reset_request(self) -> bool:
        return False

    def sync(self, base_pos_w: np.ndarray | None = None) -> None:
        del base_pos_w

    def sleep_when_paused(self) -> None:
        time.sleep(0.02)
