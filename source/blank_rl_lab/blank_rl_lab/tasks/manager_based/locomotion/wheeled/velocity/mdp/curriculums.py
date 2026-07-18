from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def force_level(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    reward_term_name: str
):
    force_command = env.command_manager.get_term("force_command")
    episode_sums = env.reward_manager._episode_sums[reward_term_name] # type: ignore
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
    if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.6 * reward_term_cfg.weight:
        force_command._command[env_ids, 0] = (force_command._command[env_ids, 0] - 10.0).clamp(min=0.0) # type: ignore
    return torch.mean(torch.squeeze(force_command.command))


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges # type: ignore
    limit_ranges = command_term.cfg.limit_ranges # type: ignore

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s # type: ignore

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)

def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges  # type: ignore
    limit_ranges = command_term.cfg.limit_ranges  # type: ignore

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = (torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s)  # type: ignore

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)

def reward_weight_schedule(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    base_weight: float,
    start_iter: int,
    end_iter: int,
    start_scale: float,
    end_scale: float,
    iteration_length: int = 24,
):
    del env_ids

    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

    current_iter = env.common_step_counter // iteration_length
    progress = (current_iter - start_iter) / max(end_iter - start_iter, 1)
    progress = min(max(progress, 0.0), 1.0)

    scale = (1.0 - progress) * start_scale + progress * end_scale
    reward_term_cfg.weight = base_weight * scale

    env.reward_manager.set_term_cfg(reward_term_name, reward_term_cfg)

    return float(reward_term_cfg.weight)


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain # type: ignore
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2 # type: ignore
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down) # type: ignore
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

def terrain_levels_by_go2_command(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
)-> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain  # type: ignore
    command_term = env.command_manager.get_term(command_name)

    current_distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )
    # max_move_distance 是跨时间累积的最大值。
    distance = torch.maximum(
        command_term.max_move_distance[env_ids],  # type: ignore
        current_distance,
    )

    terrain_length = terrain.cfg.terrain_generator.size[0]  # type: ignore

    move_up = distance > terrain_length / 2

    # 按整个 episode 采样过的 command 来看，机器人理论上被要求走多远
    # 因为 zero command 会让一部分时间不要求机器人平移，所以原版把期望距离按非 zero 概率缩一下。
    expected_distance = (
        torch.norm(command_term.commands_xy_accumulation[env_ids], dim=1)  # type: ignore
        * command_term.cfg.resampling_time_range[1]  # type: ignore
        * (1.0 - command_term.zero_command_proba)  # type: ignore
    )
    # 如果机器人实际最大移动距离还不到理论目标距离的一半，就降级。
    # 但如果已经满足 move_up，就不降级。
    move_down = (distance < expected_distance * 0.5) & ~move_up
    # move_up=True   -> terrain_levels + 1
    # move_down=True -> terrain_levels - 1
    # > 随机回到某个 level
    # 低于 0 -> clamp 到 0
    terrain.update_env_origins(env_ids, move_up, move_down) # type: ignore

    return torch.mean(terrain.terrain_levels.float())