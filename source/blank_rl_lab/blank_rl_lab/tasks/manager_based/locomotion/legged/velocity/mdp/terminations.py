from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def gravity_too_horizontal(env: 'ManagerBasedRLEnv', threshold: float=-0.1, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b[:, 2] > threshold

def extreme_parkour_route_complete(env: ManagerBasedRLEnv, command_name: str = "waypoint",) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)

    return command_term.route_complete.clone() # type:ignore