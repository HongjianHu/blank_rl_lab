"""Policy loading and inference wrappers.

The sim-to-sim runner should consume an exported inference artifact, not the
training checkpoint directly. For this project, scripts/rsl_rl/play.py exports
TorchScript and ONNX policies into the checkpoint run's exported/ directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import PolicyConfig


class BasePolicy:
    """Small runtime interface shared by TorchScript, ONNX, and zero policies."""

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        """Reset policy-side state, if any."""
        return None


class ZeroPolicy(BasePolicy):
    """Debug policy that returns zero actions.

    Use this first to check the MuJoCo model, joint mapping, default pose, and
    PD loop before introducing neural-network output.
    """

    def __init__(self, action_dim: int):
        self._action_dim = int(action_dim)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        del obs
        return np.zeros(self._action_dim, dtype=np.float32)


class TorchScriptPolicy(BasePolicy):
    """TorchScript policy wrapper."""

    def __init__(self, path: str | Path, device: str):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("TorchScript policy loading requires the 'torch' package.") from exc

        self._torch = torch
        self._device = torch.device(device)
        self._module = torch.jit.load(str(path), map_location=self._device)
        self._module.eval()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = self._torch.as_tensor(obs, dtype=self._torch.float32, device=self._device).unsqueeze(0)
        with self._torch.inference_mode():
            action = self._module(obs_tensor)
        if isinstance(action, (tuple, list)):
            action = action[0]
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def reset(self) -> None:
        if hasattr(self._module, "reset"):
            with self._torch.inference_mode():
                self._module.reset()


class OnnxPolicy(BasePolicy):
    """ONNX Runtime policy wrapper."""

    def __init__(self, path: str | Path):
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise RuntimeError("ONNX policy loading requires the 'onnxruntime' package.") from exc

        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs_batch = np.asarray(obs, dtype=np.float32)[None, :]
        action = self._session.run([self._output_name], {self._input_name: obs_batch})[0]
        return np.asarray(action[0], dtype=np.float32)


def load_policy(cfg: PolicyConfig, action_dim: int, zero_policy: bool = False) -> BasePolicy:
    """Create the configured policy wrapper."""
    if zero_policy:
        return ZeroPolicy(action_dim)

    path = Path(cfg.path)
    if not path.is_file():
        raise FileNotFoundError(f"Policy file does not exist: {path}")

    backend = cfg.backend.lower()
    if backend == "torchscript":
        return TorchScriptPolicy(path, cfg.device)
    if backend == "onnx":
        return OnnxPolicy(path)
    raise ValueError(f"Unsupported policy backend: {cfg.backend}. Expected 'torchscript' or 'onnx'.")
