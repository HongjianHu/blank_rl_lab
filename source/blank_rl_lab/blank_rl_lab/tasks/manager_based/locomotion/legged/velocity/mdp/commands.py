from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from dataclasses import MISSING
import itertools

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

class ForceCommand(CommandTerm):

    cfg: ForceCommandCfg

    def __init__(self, cfg: ForceCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env) # type:ignore

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

    force: float = MISSING # type:ignore

@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING # type:ignore

"""
env reset
  -> command_manager.reset(env_ids)
    -> term.reset(env_ids)
      -> term._resample(env_ids)
        -> term.time_left[...] = random(resampling_time_range)
        -> term._resample_command(env_ids)
        -> term.command_counter[...] += 1

each env step
  -> command_manager.compute(dt)
    -> term._update_metrics()
    -> term.time_left -= dt
    -> if time_left <= 0: term._resample(env_ids)
    -> term._update_command()

self.time_left:每个 env 距离下一次重采样 command 还剩多少秒。注意是“秒”，不是 step 数。
self.command_counter:当前 episode 内这个 command 被重采样了几次。
self.vel_command_b:IsaacLab UniformVelocityCommand 里的核心 command.shape 是 (num_envs, 3)，分别是 base frame 下的 lin_vel_x, lin_vel_y, ang_vel_z。
self.robot:由 cfg.asset_name 找到的机器人 articulation,来自 env.scene[cfg.asset_name]。
self.robot.data.root_pos_w:机器人 base/root 在世界系的位置。
env.scene.env_origins:每个并行环境的原点。rough terrain 下它通常对应每个 terrain tile 的中心/起点参考。
self._env.step_dt:RL 环境一步的时间，也就是 policy/control step,不是单个 physics substep。
"""

class GoStyleVelocityCommand(UniformVelocityCommand):
    # 这里写 go2_rl_gym 原来的 _resample_commands 逻辑
    cfg: GoStyleLevelVelocityCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # 它不是机器人实际走了多远，而是“这一轮 episode 内采样过的 xy command 的累计”。
        # 后面 terrain curriculum 降级时会用它估计“按 command 来说你本来应该走多远
        self.commands_xy_accumulation = torch.zeros(self.num_envs, 2, device=self.device)
        # 它记录当前 episode 机器人离 terrain origin 最远到过哪里。
        # terrain curriculum 升级时用它判断“你有没有走过半个 terrain tile
        self.max_move_distance = torch.zeros(self.num_envs, device=self.device)
        self.last_is_limit_vel = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.zero_command_proba = 0.0
        self.limit_vel_comb = torch.tensor(
            list(itertools.product(
            cfg.limit_vel_x,    # type:ignore
            cfg.limit_vel_y,    # type:ignore
            cfg.limit_vel_yaw,  # type:ignore
            )),
        device=self.device,
        dtype=torch.long,
        )
        self.terrain_type_names = self._build_terrain_type_names()
        self._refresh_env_command_ranges()

    def _refresh_env_command_ranges(self):
        ranges = self.cfg.ranges

        self.env_command_ranges = {
            "lin_vel_x": torch.tensor(
                ranges.lin_vel_x,
                device=self.device,
                dtype=torch.float,
            ).repeat(self.num_envs, 1),
            "lin_vel_y": torch.tensor(
                ranges.lin_vel_y,
                device=self.device,
                dtype=torch.float,
            ).repeat(self.num_envs, 1),
            "ang_vel_z": torch.tensor(
                ranges.ang_vel_z,
                device=self.device,
                dtype=torch.float,
            ).repeat(self.num_envs, 1),
        }

        self._apply_terrain_command_ranges()

    def _apply_terrain_command_ranges(self):
        if self.cfg.terrain_command_ranges is None:
           return
        if self.terrain_type_names is None:
           return
        if not hasattr(self._env.scene.terrain, "terrain_types"):
           return
        # 它是每个 env 当前所属的 terrain column。
        terrain_types = self._env.scene.terrain.terrain_types # type:ignore

        for terrain_cfg in self.cfg.terrain_command_ranges:
            terrain_names = set(terrain_cfg["terrain_names"])
            matching_cols = [
                col for col, name in enumerate(self.terrain_type_names)
                if name in terrain_names
            ]

            if len(matching_cols) == 0:
                continue

            matching_cols_tensor = torch.tensor(matching_cols, device=self.device)
            env_mask = torch.isin(terrain_types, matching_cols_tensor)
            env_ids = env_mask.nonzero(as_tuple=False).flatten()

            if len(env_ids) == 0:
               continue

            self.env_command_ranges["lin_vel_x"][env_ids, 0] = torch.maximum(
                self.env_command_ranges["lin_vel_x"][env_ids, 0],
                torch.tensor(terrain_cfg["lin_vel_x"][0], device=self.device),
            )

            self.env_command_ranges["lin_vel_x"][env_ids, 1] = torch.minimum(
                self.env_command_ranges["lin_vel_x"][env_ids, 1],
                torch.tensor(terrain_cfg["lin_vel_x"][1], device=self.device),
            )

            self.env_command_ranges["lin_vel_y"][env_ids, 0] = torch.maximum(
                self.env_command_ranges["lin_vel_y"][env_ids, 0],
                torch.tensor(terrain_cfg["lin_vel_y"][0], device=self.device),
            )
            self.env_command_ranges["lin_vel_y"][env_ids, 1] = torch.minimum(
                self.env_command_ranges["lin_vel_y"][env_ids, 1],
                torch.tensor(terrain_cfg["lin_vel_y"][1], device=self.device),
            )

            self.env_command_ranges["ang_vel_z"][env_ids, 0] = torch.maximum(
                self.env_command_ranges["ang_vel_z"][env_ids, 0],
                torch.tensor(terrain_cfg["ang_vel_z"][0], device=self.device),
            )

            self.env_command_ranges["ang_vel_z"][env_ids, 1] = torch.minimum(
                self.env_command_ranges["ang_vel_z"][env_ids, 1],
                torch.tensor(terrain_cfg["ang_vel_z"][1], device=self.device),
            )

    def _get_current_scale(self, config: dict) -> float:
        current_iter = self._env.common_step_counter // self.cfg.curriculum_iteration_length # type: ignore
        start_iter = config["start_iter"]
        end_iter = config["end_iter"]
        start_value = config["start_value"]
        end_value = config["end_value"]

        ratio = (current_iter - start_iter) / (end_iter - start_iter)
        ratio = max(min(ratio, 1.0), 0.0)

        return (1.0 - ratio) * start_value + ratio * end_value

    def _get_current_max_lin_vel(self, env_ids):
        ranges = self.env_command_ranges
        max_x = torch.maximum(
            torch.abs(ranges["lin_vel_x"][env_ids, 0]),
            torch.abs(ranges["lin_vel_x"][env_ids, 1]),
        )
        max_y = torch.maximum(
            torch.abs(ranges["lin_vel_y"][env_ids, 0]),
            torch.abs(ranges["lin_vel_y"][env_ids, 1]),
        )
        return torch.maximum(max_x, max_y).clamp_min(1e-6)

    def _get_remaining_dist(self, env_ids):
        max_command_time = self.cfg.resampling_time_range[1]
        # 如果未来变成随机持续时间，max 是偏乐观估计，min 是偏保守估计
        return torch.clamp(
            0.625 * self.cfg.terrain_length
            - torch.norm(self.commands_xy_accumulation[env_ids], dim=1) * max_command_time,
            min=0.0,
        )

    @staticmethod
    def sample_disjoint_intervals(
            lower_bound: torch.Tensor, # type:ignore
            range_min,
            range_max,
            device: str | torch.device,
        ):
        width_neg = torch.nn.functional.relu(-lower_bound - range_min)
        width_pos = torch.nn.functional.relu(range_max - lower_bound)

        total_width = width_neg + width_pos + 1e-6
        sample = torch.rand(len(lower_bound), device=device) * total_width

        return torch.where(
            sample < width_neg,
            range_min + sample,
            range_max - width_pos + (sample - width_neg),
        )

    def _build_terrain_type_names(self) -> list[str] | None:
        terrain = getattr(self._env.scene, "terrain", None)
        if terrain is None or terrain.cfg.terrain_generator is None:
           return None

        terrain_gen_cfg = terrain.cfg.terrain_generator
        sub_terrains = terrain_gen_cfg.sub_terrains
        num_cols = terrain_gen_cfg.num_cols

        proportions = torch.tensor(
            [sub_cfg.proportion for sub_cfg in sub_terrains.values()],
            device=self.device,
            dtype=torch.float,
        )
        proportions = proportions / torch.sum(proportions)
        cumulative = torch.cumsum(proportions, dim=0)

        terrain_names = list(sub_terrains.keys())
        col_to_name = []

        for col in range(num_cols):
            choice = col / num_cols + 0.001
            terrain_idx = int(torch.nonzero(choice < cumulative, as_tuple=False)[0].item())
            col_to_name.append(terrain_names[terrain_idx])
        return col_to_name


    def _limited_values_from_codes(self, codes, min_value, max_value):
        values = torch.empty(len(codes), device=self.device)

        values[codes == -1] = min_value[codes == -1]
        values[codes == 0] = 0.0
        values[codes == 1] = max_value[codes == 1]

        return values

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.commands_xy_accumulation[ids] = 0.0
        self.max_move_distance[ids] = 0.0
        self.last_is_limit_vel[ids] = False
        return super().reset(env_ids)

    def _maybe_update_command_range_curriculum(self):
        if not self.cfg.command_range_curriculum:
            return
        current_iter = self._env.common_step_counter // self.cfg.curriculum_iteration_length # type:ignore
        target_stage = None
        for stage in self.cfg.command_range_curriculum:
            if current_iter >= stage["iter"]:
               target_stage = stage

        if target_stage is None:
            return
        old_ranges = self.cfg.ranges
        new_ranges = type(old_ranges)(
        lin_vel_x=tuple(target_stage["lin_vel_x"]),
        lin_vel_y=tuple(target_stage["lin_vel_y"]),
        ang_vel_z=tuple(target_stage["ang_vel_z"]),
        heading=old_ranges.heading,
    )
        if new_ranges != old_ranges:
           self.cfg.ranges = new_ranges
           self._refresh_env_command_ranges()

    def _maybe_apply_zero_command_curriculum(self, env_ids, rand_prob, min_prob):
        if self.cfg.zero_command_curriculum is None:
            return min_prob

        self.zero_command_proba = self._get_current_scale(self.cfg.zero_command_curriculum)

        if self.zero_command_proba <= 0.0:
            return min_prob

        max_command_time = self.cfg.resampling_time_range[1]

        remaining_dist = self._get_remaining_dist(env_ids)

        remaining_episode_time = (
            self._env.max_episode_length - self._env.episode_length_buf[env_ids]) * self._env.step_dt # type:ignore

        max_lin_vel = self._get_current_max_lin_vel(env_ids)

        next_time = (
            remaining_episode_time
            - remaining_dist / (0.8 * max_lin_vel + 1e-9)
        ).clamp(
            min = 0.0,
            max = max_command_time,
        )

        max_prob = min_prob + self.zero_command_proba

        zero_mask = (
            (rand_prob >= min_prob)
            & (rand_prob < max_prob)
            & (next_time > 0.0)
        )

        zero_env_ids = env_ids[zero_mask]

        if len(zero_env_ids) == 0:
            return max_prob

        self.vel_command_b[zero_env_ids, :2] = 0.0
        self.time_left[zero_env_ids] = next_time[zero_mask]

        if self.cfg.limit_ang_vel_at_zero_command_prob > 0.0:
           add_ang_mask = torch.rand(len(zero_env_ids), device=self.device) < self.cfg.limit_ang_vel_at_zero_command_prob
           add_ang_env_ids = zero_env_ids[add_ang_mask]

           if len(add_ang_env_ids) > 0:
               choose_min = torch.rand(len(add_ang_env_ids), device=self.device) < 0.5
               yaw_min = self.env_command_ranges["ang_vel_z"][add_ang_env_ids, 0]
               yaw_max = self.env_command_ranges["ang_vel_z"][add_ang_env_ids, 1]
               self.vel_command_b[add_ang_env_ids, 2] = torch.where(choose_min, yaw_min, yaw_max)
        return max_prob

    def _maybe_apply_limit_velocity(self, env_ids, rand_prob, min_prob: float):
        if self.cfg.limit_vel_prob <= 0.0:
            return min_prob

        max_prob = min_prob + self.cfg.limit_vel_prob
        limit_mask = (rand_prob >= min_prob) & (rand_prob < max_prob)
        limit_env_ids = env_ids[limit_mask]

        if len(limit_env_ids) == 0:
            self.last_is_limit_vel[env_ids] = False
            return max_prob

        change_limit_env_ids = limit_env_ids

        if self.cfg.limit_vel_invert_when_continuous:
           was_limited = self.last_is_limit_vel[limit_env_ids]

           invert_env_ids = limit_env_ids[was_limited]
           if len(invert_env_ids) > 0:
              self.vel_command_b[invert_env_ids, :] *= -1.0

           change_limit_env_ids = limit_env_ids[~was_limited]

        if len(change_limit_env_ids) > 0:
            combo_ids = torch.randint(
                0,
                self.limit_vel_comb.shape[0],
                (len(change_limit_env_ids),),
                device=self.device,
            )
            combos = self.limit_vel_comb[combo_ids]

            ranges = self.env_command_ranges

            self.vel_command_b[change_limit_env_ids, 0] = self._limited_values_from_codes(
                combos[:, 0],
                ranges["lin_vel_x"][change_limit_env_ids, 0],
                ranges["lin_vel_x"][change_limit_env_ids, 1],
            )
            self.vel_command_b[change_limit_env_ids, 1] = self._limited_values_from_codes(
                combos[:, 1],
                ranges["lin_vel_y"][change_limit_env_ids, 0],
                ranges["lin_vel_y"][change_limit_env_ids, 1],
            )
            self.vel_command_b[change_limit_env_ids, 2] = self._limited_values_from_codes(
                combos[:, 2],
                ranges["ang_vel_z"][change_limit_env_ids, 0],
                ranges["ang_vel_z"][change_limit_env_ids, 1],
            )
        self.last_is_limit_vel[env_ids] = False
        self.last_is_limit_vel[limit_env_ids] = True

        return max_prob


    def _resample_dynamic_command(self, env_ids):
        remaining_dist = self._get_remaining_dist(env_ids)

        remaining_episode_time = (
            self._env.max_episode_length - self._env.episode_length_buf[env_ids]) * self._env.step_dt # type:ignore

        vel_low_bound = torch.clamp(
            remaining_dist / (remaining_episode_time + 1e-9),
            min=0.0,
        )

        ranges = self.env_command_ranges

        self.vel_command_b[env_ids, 0] = self.sample_disjoint_intervals(
            vel_low_bound,
            ranges["lin_vel_x"][env_ids, 0],
            ranges["lin_vel_x"][env_ids, 1],
            self.device,
        )

        self.vel_command_b[env_ids, 1] = self.sample_disjoint_intervals(
            vel_low_bound,
            ranges["lin_vel_y"][env_ids, 0],
            ranges["lin_vel_y"][env_ids, 1],
            self.device,
        )

        n = len(env_ids)
        r = torch.rand(n, device=self.device)
        self.vel_command_b[env_ids, 2] = (
            ranges["ang_vel_z"][env_ids, 1] - ranges["ang_vel_z"][env_ids, 0]
        ) * r + ranges["ang_vel_z"][env_ids, 0]

        if self.cfg.heading_command:
            heading_range = self.cfg.ranges.heading
            r = torch.rand(n, device=self.device)
            self.heading_target[env_ids] = (
                heading_range[1] - heading_range[0] # type:ignore
            ) * r + heading_range[0] # type:ignore
            self.is_heading_env[env_ids] = torch.rand(n, device=self.device) <= self.cfg.rel_heading_envs

        self.is_standing_env[env_ids] = torch.rand(n, device=self.device) <= self.cfg.rel_standing_envs


    def _resample_uniform_command(self, env_ids):
        ranges = self.env_command_ranges
        n = len(env_ids)

        r = torch.rand(n, device=self.device)
        self.vel_command_b[env_ids, 0] = (
            ranges["lin_vel_x"][env_ids, 1] - ranges["lin_vel_x"][env_ids, 0]
        ) * r + ranges["lin_vel_x"][env_ids, 0]

        r = torch.rand(n, device=self.device)
        self.vel_command_b[env_ids, 1] = (
            ranges["lin_vel_y"][env_ids, 1] - ranges["lin_vel_y"][env_ids, 0]
        ) * r + ranges["lin_vel_y"][env_ids, 0]

        r = torch.rand(n, device=self.device)
        self.vel_command_b[env_ids, 2] = (
            ranges["ang_vel_z"][env_ids, 1] - ranges["ang_vel_z"][env_ids, 0]
        ) * r + ranges["ang_vel_z"][env_ids, 0]

        self.is_standing_env[env_ids] = torch.rand(n, device=self.device) <= self.cfg.rel_standing_envs

        if self.cfg.heading_command:
            r = torch.rand(n, device=self.device)
            heading_range = self.cfg.ranges.heading
            self.heading_target[env_ids] = (
                heading_range[1] - heading_range[0]) * r + heading_range[0] # type:ignore
            self.is_heading_env[env_ids] = torch.rand(n, device=self.device) <= self.cfg.rel_heading_envs
        else:
            r = torch.rand(n, device=self.device)
            self.vel_command_b[env_ids, 2] = (
                ranges["ang_vel_z"][env_ids, 1] - ranges["ang_vel_z"][env_ids, 0]
            ) * r + ranges["ang_vel_z"][env_ids, 0]

    def _resample_command(self, env_ids):
        # 先更新 command range
        self._maybe_update_command_range_curriculum()
        if self.cfg.dynamic_resample_commands:
            # 动态采样 command
           self._resample_dynamic_command(env_ids)
        else:
            # 普通采样 command
            self._resample_uniform_command(env_ids)

        rand_prob = torch.rand(len(env_ids), device=self.device)
        prob_cursor = 0.0
        # limit_vel 覆盖一部分 env (limit这里指的是极限而不是限制)
        prob_cursor = self._maybe_apply_limit_velocity(env_ids, rand_prob, prob_cursor)
        # 再按概率把部分 env 的 xy command 置零
        prob_cursor = self._maybe_apply_zero_command_curriculum(env_ids, rand_prob, prob_cursor)
        # 最后累计 commands_xy_accumulation
        self.commands_xy_accumulation[env_ids] += self.vel_command_b[env_ids, :2]

    # 是在decimation个物理步结束后，由command_manager.compute(dt=self.step_dt)调一次
    def _update_command(self):
        super()._update_command()

        dist = torch.norm(
            self.robot.data.root_pos_w[:, :2] - self._env.scene.env_origins[:, :2],
            dim=1,
        )
        self.max_move_distance = torch.maximum(self.max_move_distance, dist)


@configclass
class GoStyleLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = GoStyleVelocityCommand

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING # type:ignore

    command_range_curriculum: list[dict] | None = None
    curriculum_iteration_length: int = 24
    terrain_command_ranges: list[dict] | None = None

    zero_command_curriculum: dict | None = None
    limit_ang_vel_at_zero_command_prob: float = 0.0
    terrain_length: float = 8.0

    dynamic_resample_commands: bool = False
    limit_vel_prob: float = 0.0
    limit_vel_x: tuple[int, ...] = (-1, 1)
    limit_vel_y: tuple[int, ...] = (-1, 1)
    limit_vel_yaw: tuple[int, ...] = (-1, 0, 1)
    limit_vel_invert_when_continuous: bool = True


class ExtremeParkourCommand(CommandTerm):
    """Waypoint navigation command for the Extreme Parkour terrain."""

    cfg: ExtremeParkourCommandCfg

    def __init__(self, cfg: ExtremeParkourCommandCfg, env: ManagerBasedRLEnv,) -> None:
        super().__init__(cfg, env)

        if cfg.num_waypoints <= 1:
            raise ValueError(
                f"num_waypoints must be greater than one, got {cfg.num_waypoints}."
            )
        if cfg.reach_threshold <= 0.0:
            raise ValueError(
                f"reach_threshold must be positive, got {cfg.reach_threshold}."
            )
        if cfg.reach_hold_time <= 0.0:
            raise ValueError(
                f"reach_hold_time must be positive, got {cfg.reach_hold_time}."
            )
        if cfg.forward_speed_range[0] > cfg.forward_speed_range[1]:
            raise ValueError(
                "forward_speed_range lower bound must not exceed upper bound: "
                f"{cfg.forward_speed_range}."
            )

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.terrain = env.scene.terrain

        required_terrain_attributes = (
            "waypoints_grid",
            "terrain_class_grid",
            "difficulty_grid",
            "terrain_levels",
            "terrain_types",
        )

        missing_attributes = [
            name
            for name in required_terrain_attributes
            if not hasattr(self.terrain, name)
        ]
        if missing_attributes:
            raise RuntimeError(
                "ExtremeParkourCommand requires metadata-aware terrain. "
                f"Missing terrain attributes: {missing_attributes}."
            )

        waypoint_grid = self.terrain.waypoints_grid # type:ignore

        if waypoint_grid.ndim != 4:
            raise RuntimeError(
                "waypoints_grid must have shape "
                "[num_rows, num_cols, num_waypoints, 3], "
                f"got {tuple(waypoint_grid.shape)}."
            )
        if waypoint_grid.shape[2] != cfg.num_waypoints:
            raise RuntimeError(
                f"Expected {cfg.num_waypoints} waypoints per route, "
                f"got {waypoint_grid.shape[2]}."
            )
        if waypoint_grid.shape[3] != 3:
            raise RuntimeError(
                "Each waypoint must contain xyz coordinates, "
                f"got final dimension {waypoint_grid.shape[3]}."
            )

        self.route_waypoints_w = torch.zeros(
            self.num_envs,
            cfg.num_waypoints,
            3,
            device=self.device,
            dtype=torch.float32,
        )

        # Index range is [0, num_waypoints].
        # num_waypoints means that all waypoints have been completed.
        self.waypoint_index = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )

        self.reach_time = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float32,
        )

        self.route_complete = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )

        # Current and next waypoint positions in world coordinates.
        self.current_waypoint_w = torch.zeros(
            self.num_envs,
            3,
            device=self.device,
        )

        self.next_waypoint_w = torch.zeros_like(
            self.current_waypoint_w
        )

        # Vectors from the robot to the waypoint in world xy coordinates.
        self.target_pos_rel_w = torch.zeros(
            self.num_envs,
            2,
            device=self.device,
        )
        self.next_target_pos_rel_w = torch.zeros_like(
            self.target_pos_rel_w
        )

        self.target_direction_w = torch.zeros_like(
            self.target_pos_rel_w
        )

        self.next_target_direction_w = torch.zeros_like(
            self.next_target_pos_rel_w
        )

        self.distance_to_waypoint = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        # World-frame target headings.
        self.target_yaw_w = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.next_target_yaw_w = torch.zeros_like(
            self.target_yaw_w
        )

        # Heading errors relative to the Go2 base heading.
        self.heading_error = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.next_heading_error = torch.zeros_like(
            self.heading_error
        )

        self.forward_speed = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        self.terrain_level = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        self.terrain_type = torch.zeros_like(
            self.terrain_level
        )
        self.terrain_class = torch.zeros_like(
            self.terrain_level
        )
        self.difficulty = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        self._command = torch.zeros(
            self.num_envs,
            8,
            device=self.device,
            dtype=torch.float32,
        )

        self.metrics["distance_to_waypoint"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["waypoint_progress"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["route_complete"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["terrain_level"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )
        self.metrics["difficulty"] = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        all_env_ids = torch.arange(
            self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        self._resample_command(all_env_ids) # type:ignore
        self.time_left[:] = cfg.resampling_time_range[1]

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _synchronize_route_assignment(self) -> None:
        """Reload routes whose terrain level or terrain type has changed."""
        current_levels = self.terrain.terrain_levels # type:ignore
        current_types = self.terrain.terrain_types   # type:ignore
        # curriculum change terrain_levels, thus terrain_type change
        assignment_changed = (self.terrain_level != current_levels) | (self.terrain_type != current_types)

        changed_env_ids = assignment_changed.nonzero(as_tuple=False).flatten()

        if changed_env_ids.numel() > 0:
            self._resample_command(changed_env_ids) #type:ignore


    def _refresh_waypoint_buffers(self) -> None:
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long,)

        current_indices = torch.clamp(self.waypoint_index, min=0, max=self.cfg.num_waypoints - 1,)

        next_indices = torch.clamp(current_indices + 1, min=0, max=self.cfg.num_waypoints - 1,)

        self.current_waypoint_w[:] = self.route_waypoints_w[env_ids, current_indices,]

        self.next_waypoint_w[:] = self.route_waypoints_w[env_ids, next_indices,]

    def _update_navigation_geometry(self) -> None:

        root_xy_w = self.robot.data.root_pos_w[:, :2]

        self.target_pos_rel_w[:] = self.current_waypoint_w[:, :2] - root_xy_w

        self.next_target_pos_rel_w[:] = self.next_waypoint_w[:, :2] - root_xy_w

        self.distance_to_waypoint[:] = torch.linalg.norm(self.target_pos_rel_w, dim = 1)

        target_norm = torch.linalg.vector_norm(self.target_pos_rel_w, dim=1, keepdim=True,)

        next_target_norm = torch.linalg.vector_norm(self.next_target_pos_rel_w, dim=1, keepdim=True,)

        self.target_direction_w[:] = self.target_pos_rel_w / target_norm.clamp_min(1.0e-6)

        self.next_target_direction_w[:] = self.next_target_pos_rel_w / next_target_norm.clamp_min(1.0e-6)

        self.target_yaw_w[:] = torch.atan2(self.target_direction_w[:, 1], self.target_direction_w[:, 0])

        self.next_target_yaw_w[:] = torch.atan2(self.next_target_direction_w[:, 1], self.next_target_direction_w[:, 0])

        robot_yaw_w = self.robot.data.heading_w

        self.heading_error[:] = math_utils.wrap_to_pi(self.target_yaw_w - robot_yaw_w)

        self.next_heading_error[:] = math_utils.wrap_to_pi(self.next_target_yaw_w- robot_yaw_w)

    def _write_command_tensor(self) -> None:
        self._command[:, 0] = 0.0
        self._command[:, 1] = self.heading_error
        self._command[:, 2] = self.next_heading_error
        self._command[:, 3:5] = 0.0
        self._command[:, 5] = self.forward_speed

        is_demo = (self.terrain_class == self.cfg.parkour_flat_terrain_class)
        self._command[:, 6] = (~is_demo).float()
        self._command[:, 7] = is_demo.float()

    def _resample_command(self, env_ids: Sequence[int],) -> None:
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long,) # type:ignore
        if env_ids.numel() == 0: #type:ignore
            return

        # self.terrain.waypoints_grid: [num_rows, num_cols, num_waypoints, xyz]
        # self.terrain.terrain_levels[env_ids] 表示每个并行环境当前被分配到第几行地形
        # self.terrain.terrain_types[env_ids] 表示实际表示地形网格的列编号
        terrain_levels = self.terrain.terrain_levels[env_ids] # type:ignore
        terrain_types = self.terrain.terrain_types[env_ids]   # type:ignore

        # Save the terrain assignment corresponding to the cached route.
        self.terrain_level[env_ids] = terrain_levels
        self.terrain_type[env_ids] = terrain_types

        self.route_waypoints_w[env_ids] = (self.terrain.waypoints_grid[terrain_levels,terrain_types,])      # type:ignore
        # terrain_class：这一列属于五大类中的哪一类，范围 0～4                                                    # type:ignore
        self.terrain_class[env_ids] = (self.terrain.terrain_class_grid[terrain_levels, terrain_types,])     # type:ignore
        self.difficulty[env_ids] = (self.terrain.difficulty_grid[terrain_levels, terrain_types,])   # type:ignore

        self.waypoint_index[env_ids] = 0
        self.reach_time[env_ids] = 0.0
        self.route_complete[env_ids] = False

        speed_min, speed_max = self.cfg.forward_speed_range
        self.forward_speed[env_ids] = (speed_min + (speed_max - speed_min) * torch.rand(len(env_ids), device=self.device))

        self._refresh_waypoint_buffers()
        self._update_navigation_geometry()
        self._write_command_tensor()

    def _update_command(self) -> None:
        """Advance waypoint indices after a continuous hold inside the target."""
        self._synchronize_route_assignment()
        self._refresh_waypoint_buffers()
        self._update_navigation_geometry()

        active = ~self.route_complete
        inside_target =  self.distance_to_waypoint < self.cfg.reach_threshold

        holding_target = active & inside_target
        # Resetting to zero whenever the robot leaves the target radius makes
        # reach_hold_time a continuous hold rather than accumulated visits.
        self.reach_time[:] = torch.where(holding_target, self.reach_time + self._env.step_dt, torch.zeros_like(self.reach_time),)

        advance = active & (self.reach_time >= self.cfg.reach_hold_time - 1.0e-6)

        if torch.any(advance):
           self.waypoint_index[advance] += 1
           # Waypoint == 8的含义是 8 个 waypoint 已全部完成
           self.waypoint_index.clamp_(max=self.cfg.num_waypoints)
           self.reach_time[advance] = 0.0
           self.route_complete[:] = self.waypoint_index >= self.cfg.num_waypoints

           # Immediately expose the new target in the same policy step.
           self._refresh_waypoint_buffers()
           self._update_navigation_geometry()

        self._write_command_tensor()

    def _update_metrics(self) -> None:
        """Update values logged when an environment resets."""

        self.metrics["distance_to_waypoint"][:] = (
            self.distance_to_waypoint
        )
        self.metrics["waypoint_progress"][:] = (
            self.waypoint_index.float()
            / float(self.cfg.num_waypoints)
        )
        self.metrics["route_complete"][:] = (
            self.route_complete.float()
        )
        self.metrics["terrain_level"][:] = (
            self.terrain_level.float()
        )
        self.metrics["difficulty"][:] = self.difficulty

    def _set_debug_vis_impl(self, debug_vis: bool,) -> None:
        """Create waypoint markers and toggle their visibility."""

        if debug_vis:
            if not hasattr(self, "current_waypoint_visualizer",):
                self.current_waypoint_visualizer = (
                    VisualizationMarkers(
                        self.cfg.current_waypoint_visualizer_cfg
                    )
                )
                self.next_waypoint_visualizer = (
                    VisualizationMarkers(
                        self.cfg.next_waypoint_visualizer_cfg
                    )
                )

            self.current_waypoint_visualizer.set_visibility(True)
            self.next_waypoint_visualizer.set_visibility(True)

        else:
            if hasattr(self, "current_waypoint_visualizer",):
                self.current_waypoint_visualizer.set_visibility(False)
                self.next_waypoint_visualizer.set_visibility(False)

    # 启用调试显示后，IsaacLab 会将该方法订阅到仿真应用的 post-update event。
    def _debug_vis_callback(self, event,) -> None:
        """Update current and next waypoint markers."""

        del event

        if not self.robot.is_initialized:
            return

        current_marker_positions = self.current_waypoint_w.clone()

        next_marker_positions = self.next_waypoint_w.clone()

        current_marker_positions[:, 2] += self.cfg.marker_height_offset

        next_marker_positions[:, 2] += self.cfg.marker_height_offset

        current_marker_scales = torch.ones(self.num_envs, 3, device=self.device,)

        current_marker_scales[self.route_complete] = 0.0

        has_next_waypoint = (
            ~self.route_complete
            & (self.waypoint_index < self.cfg.num_waypoints - 1)
        )

        next_marker_scales = torch.zeros(self.num_envs, 3, device=self.device,)

        next_marker_scales[has_next_waypoint] = 1.0

        self.current_waypoint_visualizer.visualize(
            translations=current_marker_positions,
            scales=current_marker_scales,
        )

        self.next_waypoint_visualizer.visualize(
            translations=next_marker_positions,
            scales=next_marker_scales,
        )

@configclass
class ExtremeParkourCommandCfg(CommandTermCfg):
    """Configuration for the Extreme Parkour waypoint command."""

    class_type: type = ExtremeParkourCommand

    # Route changes are controlled by episode reset and waypoint switching,
    # not by CommandTerm's periodic resampling timer.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9,)

    asset_name: str = "robot"
    num_waypoints: int = 8

    reach_threshold: float = 0.2
    reach_hold_time: float = 0.1

    forward_speed_range: tuple[float, float] = (
        0.3,
        0.8,
    )

    parkour_flat_terrain_class: int = 2

    marker_height_offset: float = 0.12

    current_waypoint_visualizer_cfg: VisualizationMarkersCfg = (
    VisualizationMarkersCfg(
        prim_path=(
            "/Visuals/Command/"
            "extreme_parkour_current_waypoint"
        ),
        markers={
            "sphere": sim_utils.SphereCfg( # type:ignore
                radius=0.10,
                visual_material=sim_utils.PreviewSurfaceCfg( # type:ignore
                    diffuse_color=(0.0, 1.0, 0.0),
                    roughness=0.8,
                        ),
                    ),
                },
            )
        )

    next_waypoint_visualizer_cfg: VisualizationMarkersCfg = (
    VisualizationMarkersCfg(
        prim_path=(
            "/Visuals/Command/"
            "extreme_parkour_next_waypoint"
        ),
        markers={
            "sphere": sim_utils.SphereCfg( # type:ignore
                radius=0.07,
                visual_material=sim_utils.PreviewSurfaceCfg( # type:ignore
                    diffuse_color=(0.0, 0.35, 1.0),
                    roughness=0.8,
                        ),
                    ),
                },
            )
        )
