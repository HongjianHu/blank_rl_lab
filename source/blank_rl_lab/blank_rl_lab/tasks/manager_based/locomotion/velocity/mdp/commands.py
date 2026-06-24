from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from dataclasses import MISSING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.envs.mdp import UniformVelocityCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

class ForceCommand(CommandTerm):

    cfg: ForceCommandCfg

    def __init__(self, cfg: ForceCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._command = torch.zeros(self.num_envs, 1, device=self.device)
        self._command[:, 0] = cfg.force

    def __str__(self) -> str:
        msg = "ForceCommand: \n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass

@configclass
class ForceCommandCfg(CommandTermCfg):

    class_type: type = ForceCommand

    force: float = MISSING

@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
