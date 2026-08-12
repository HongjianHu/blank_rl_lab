from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Any, cast

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg, ActionManager
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg, RewardTermCfg
from isaaclab.assets import Articulation
from isaaclab.sensors import RayCaster, ContactSensor
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ang_vel_xy(
    env: ManagerBasedRLEnv, target_base_height_phase3: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_link_pos_w[:, 2] > target_base_height_phase3
    return torch.exp(torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1) * -2.0) * base_height


def lin_vel_xy(
    env: ManagerBasedRLEnv, target_base_height_phase3: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_link_pos_w[:, 2] > target_base_height_phase3
    return torch.exp(torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1) * -5.0) * base_height


def target_orientation(
    env: ManagerBasedRLEnv, target_base_height_phase3: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    standup = asset.data.root_link_pos_w[:, 2] > target_base_height_phase3
    return torch.exp(torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) * -5) * standup


def target_base_height(
    env: ManagerBasedRLEnv, base_height_target: float, target_base_height_phase3: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_link_pos_w[:, 2]
    standup = base_height > target_base_height_phase3
    return torch.exp(torch.abs(base_height - base_height_target) * -20.0) * standup


def target_joint_deviation_l2(
    env: ManagerBasedRLEnv, target_base_height_phase3: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    standup = asset.data.root_link_pos_w[:, 2] > target_base_height_phase3
    return torch.sum(torch.square(angle), dim=1) * standup

def energy_new_actual(env, asset_cfg=SceneEntityCfg("robot"), sigma_lin=1000.0, sigma_ang=500.0, clip_lin=0.2, clip_ang=0.2):
    asset = env.scene[asset_cfg.name]
    # 关节速度
    joint_vel = asset.data.joint_vel
    # 力矩
    joint_torque = asset.data.applied_torque
    # 这一部分算的是分子
    energy = torch.sum(torch.abs(joint_vel * joint_torque), dim=1)
    base_lin_vel_x = asset.data.root_lin_vel_b[:, 0]
    base_ang_vel_z = asset.data.root_ang_vel_b[:, 2]
    # 这一部分算的是分母
    denom = (
        sigma_lin * torch.clamp(torch.abs(base_lin_vel_x), min=clip_lin)
        + sigma_ang * torch.clamp(torch.abs(base_ang_vel_z), min=clip_ang)
    )
    # 返回能量奖励
    return torch.exp(-energy / denom)

def feet_slip(env, sensor_cfg, asset_cfg=SceneEntityCfg("robot")):
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    # [num_envs, history_length, num_bodies, 3] → [num_envs, history_length, K, 3]
    # [num_envs, history_length, K, 3] → [num_envs, history_length, K]
    # [num_envs, history_length, K] → [num_envs, K]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(contacts * torch.sum(torch.square(feet_vel_xy), dim=-1), dim=1)


def safe_base_height_l2(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    max_abs_error: float = 1.0,
):
    """Base-height penalty that ignores invalid ray hits on rough terrain."""
    asset = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    if sensor_cfg is None:
        adjusted_target_height = torch.full_like(base_height, target_height)
    else:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        ray_heights = sensor.data.ray_hits_w[..., 2]
        finite_hits = torch.isfinite(ray_heights)
        safe_hits = torch.where(finite_hits, ray_heights, torch.zeros_like(ray_heights))
        hit_count = finite_hits.sum(dim=1).clamp(min=1)
        mean_hit_height = safe_hits.sum(dim=1) / hit_count
        fallback_height = base_height - target_height
        mean_hit_height = torch.where(finite_hits.any(dim=1), mean_hit_height, fallback_height)
        adjusted_target_height = target_height + mean_hit_height
    height_error = torch.clamp(base_height - adjusted_target_height, min=-max_abs_error, max=max_abs_error)
    return torch.square(height_error)

def action_smoothness_second_order(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",  # action term name，按你的env配置修改
) -> torch.Tensor:
    """
    Second-order action smoothness penalty.

    Penalizes the second-order finite difference of actions:
        penalty = |a_t - 2 * a_{t-1} + a_{t-2}|^2

    Requires the action history of at least 2 previous steps.
    Uses env.action_manager to access current and previous actions.
    """
    action_manager = cast(ActionManager, env.action_manager)

    # 当前 action: a_t
    action_t = action_manager.action  # shape: (num_envs, action_dim)

    # 上一步 action: a_{t-1}
    action_t1 = action_manager.prev_action  # shape: (num_envs, action_dim)

    # 上上步 action: a_{t-2}
    # IsaacLab 默认只缓存一步 prev_action，需要自行维护 a_{t-2}
    if not hasattr(env, "_action_t2"):
        # 初始化时用 a_{t-1} 填充，避免第一步产生过大惩罚
        setattr(env, "_action_t2", action_t1.clone())

    action_t2 = getattr(env, "_action_t2")  # shape: (num_envs, action_dim)
    if action_t2.shape != action_t1.shape:
        action_t2 = action_t1.clone()

    # 新 episode 的前一两步不能沿用上一个 episode 的 a_{t-2}。
    if hasattr(env, "episode_length_buf"):
        fresh_envs = env.episode_length_buf <= 1
        if torch.any(fresh_envs):
            action_t2 = action_t2.clone()
            action_t2[fresh_envs] = action_t1[fresh_envs]

    # 计算二阶差分
    second_order_diff = action_t - 2.0 * action_t1 + action_t2  # (num_envs, action_dim)

    # 更新 buffer：将 a_{t-1} 存为下一步的 a_{t-2}
    setattr(env, "_action_t2", action_t1.clone())

    # 返回每个 env 的惩罚标量（对 action_dim 求和）
    penalty = torch.sum(second_order_diff ** 2, dim=-1)  # shape: (num_envs,)

    return penalty

def track_lin_vel_xy_yaw_frame_heading_exp(env: ManagerBasedRLEnv, std: float, command_name: str, y_error_weight: float=2.0, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    cmd_term = env.command_manager.get_term(command_name)
    if hasattr(cmd_term, 'heading_target'):
        heading_err = wrap_to_pi(cmd_term.heading_target - asset.data.heading_w) #type:ignore
        heading_coef = (1.0 + torch.cos(heading_err)) * 0.5
    else:
        heading_coef = torch.ones(env.num_envs, device=env.device)
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    err_x = torch.square(cmd[:, 0] - vel_yaw[:, 0])
    err_y = y_error_weight * torch.square(cmd[:, 1] - vel_yaw[:, 1])
    return torch.exp(-(err_x + err_y) / std ** 2) * heading_coef

def dof_power_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]
    omega = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(tau * omega), dim=-1)

def foot_clearance_target(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, target_height: float=0.08, foot_offset: float=0.022, sigma: float=0.01, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name] # type:ignore
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    feet_vel_norm = torch.linalg.norm(feet_vel_xy, dim=-1)
    terrain_z_max = sensor.data.ray_hits_w[..., 2].max(dim=-1).values
    height_above = feet_z - terrain_z_max.unsqueeze(-1) - target_height - foot_offset
    err = torch.sum(feet_vel_norm * torch.square(height_above), dim=-1)
    return torch.exp(-err / sigma)

def feet_contact_stand_still(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, cmd_threshold: float=0.2, force_threshold: float=10.0) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] #type:ignore
    history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :] #type:ignore
    feet_max_force_z = torch.max(torch.abs(history[..., 2]), dim=1)[0]
    in_contact = feet_max_force_z > force_threshold
    all_in_contact = torch.all(in_contact, dim=-1)
    cmd = env.command_manager.get_command(command_name)[:, :3]
    standing = torch.linalg.norm(cmd, dim=-1) < cmd_threshold
    return (standing & all_in_contact).float()

def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, ratio: float=4.0) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] #type:ignore
    history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :] #type:ignore
    lateral_t = torch.norm(history[..., :2], dim=-1)
    vertical_t = torch.abs(history[..., 2])
    lateral_max = torch.max(lateral_t, dim=1)[0]
    vertical_max = torch.max(vertical_t, dim=1)[0]
    return torch.any(lateral_max > ratio * vertical_max, dim=1).float()

def hip_pos_deviation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    err = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(err), dim=-1)

#=======================CTS-MOSE=====================================#
def _go2_estimated_ground_z_from_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    target_height: float,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name] # type:ignore

    base_z = asset.data.root_pos_w[:, 2]
    ray_heights = sensor.data.ray_hits_w[..., 2]

    ray_xy = sensor.ray_starts[0, :, :2]
    local_mask = (
        (ray_xy[:, 0] >= x_bounds[0])
        & (ray_xy[:, 0] <= x_bounds[1])
        & (ray_xy[:, 1] >= y_bounds[0])
        & (ray_xy[:, 1] <= y_bounds[1])
    )

    valid_hits = torch.isfinite(ray_heights) & local_mask.unsqueeze(0)
    safe_heights = torch.where(valid_hits, ray_heights, torch.zeros_like(ray_heights))

    hit_count = valid_hits.sum(dim=1).clamp(min=1)
    ground_z = safe_heights.sum(dim=1) / hit_count

    fallback_ground_z = base_z - target_height
    ground_z = torch.where(valid_hits.any(dim=1), ground_z, fallback_ground_z)
    return ground_z


def go2_correct_base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_bounds: tuple[float, float] = (-0.2, 0.2),
    y_bounds: tuple[float, float] = (-0.15, 0.15),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ground_z = _go2_estimated_ground_z_from_scan(env, sensor_cfg, asset_cfg, x_bounds, y_bounds, target_height)
    base_height = asset.data.root_pos_w[:, 2] - ground_z
    return torch.square(base_height - target_height)


def go2_feet_regulation(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    x_bounds: tuple[float, float] = (-0.2, 0.2),
    y_bounds: tuple[float, float] = (-0.15, 0.15),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ground_z = _go2_estimated_ground_z_from_scan(env, sensor_cfg, SceneEntityCfg("robot"), x_bounds, y_bounds, target_height)

    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    feet_height = torch.clamp(feet_z - ground_z.unsqueeze(-1), min=0.0)

    return torch.sum(
        torch.sum(torch.square(feet_vel_xy), dim=-1)
        * torch.exp(-feet_height / (0.025 * target_height)),
        dim=-1,
    )


def go2_hip_to_default_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    hip_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(hip_error), dim=-1)

def _go2_build_terrain_type_names(env: ManagerBasedRLEnv) -> list[str] | None:
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or terrain.cfg.terrain_generator is None:
        return None

    cache_name = "_go2_dynamic_sigma_terrain_type_names"
    if hasattr(env, cache_name):
        return getattr(env, cache_name)

    terrain_gen_cfg = terrain.cfg.terrain_generator
    sub_terrains = terrain_gen_cfg.sub_terrains
    num_cols = terrain_gen_cfg.num_cols

    proportions = torch.tensor(
        [sub_cfg.proportion for sub_cfg in sub_terrains.values()],
        device=env.device,
        dtype=torch.float,
    )
    proportions = proportions / torch.sum(proportions)
    cumulative = torch.cumsum(proportions, dim=0)

    terrain_names = list(sub_terrains.keys())
    col_to_name: list[str] = []

    for col in range(num_cols):
        choice = col / num_cols + 0.001
        terrain_idx = int(torch.nonzero(choice < cumulative, as_tuple=False)[0].item())
        col_to_name.append(terrain_names[terrain_idx])

    setattr(env, cache_name, col_to_name)
    return col_to_name


def _go2_dynamic_sigma(
    env: ManagerBasedRLEnv,
    target_vel_abs: torch.Tensor,
    v_min: float,
    v_max: float,
    default_sigma: float,
    max_sigma_by_terrain: dict[str, float],
) -> torch.Tensor:
    sigma = torch.full_like(target_vel_abs, default_sigma)

    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_types") or not hasattr(terrain, "terrain_levels"):
        return sigma

    col_to_name = _go2_build_terrain_type_names(env)
    if col_to_name is None:
        return sigma

    max_sigma_by_col = torch.tensor(
        [max_sigma_by_terrain.get(name, default_sigma) for name in col_to_name],
        device=env.device,
        dtype=torch.float,
    )

    terrain_types = terrain.terrain_types.clamp(max=len(col_to_name) - 1)

    target_sigmas = max_sigma_by_col[terrain_types]

    interp_mask = (target_vel_abs >= v_min) & (target_vel_abs < v_max)
    if torch.any(interp_mask):
        ratio = (target_vel_abs[interp_mask] - v_min) / (v_max - v_min)
        sigma[interp_mask] = default_sigma + ratio * (target_sigmas[interp_mask] - default_sigma)

    max_mask = target_vel_abs >= v_max
    if torch.any(max_mask):
        sigma[max_mask] = target_sigmas[max_mask]

    # 表示 terrain level 越高，sigma 放宽越明显。低 level 不会一下子把 tracking reward 放得太松。
    level_scale = torch.clamp(torch.exp((terrain.terrain_levels.float() + 1.0) / 10.0) - 1.0, max=1.0)

    sigma = default_sigma + level_scale * (sigma - default_sigma)

    return sigma


_GO2_MAX_TRACKING_SIGMA_BY_TERRAIN = {
    "wave": 5.0 / 12.0,
    "slope": 1.0 / 4.0,
    "slope_inv": 1.0 / 4.0,
    "rough_slope": 1.0 / 4.0,
    "stairs_up": 1.0 / 2.0,
    "stairs_down": 1.0 / 2.0,
    "obstacles": 3.0 / 4.0,
    "flat": 1.0 / 4.0,
}


def go2_track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Legacy Go2 XY tracking reward with command- and terrain-dependent sigma."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    sigma_x = _go2_dynamic_sigma(
        env,
        torch.abs(command[:, 0]),
        v_min=0.5,
        v_max=1.5,
        default_sigma=0.25,
        max_sigma_by_terrain=_GO2_MAX_TRACKING_SIGMA_BY_TERRAIN,
    )
    sigma_y = _go2_dynamic_sigma(
        env,
        torch.abs(command[:, 1]),
        v_min=0.5,
        v_max=1.5,
        default_sigma=0.25,
        max_sigma_by_terrain=_GO2_MAX_TRACKING_SIGMA_BY_TERRAIN,
    )

    error_sq = torch.square(command[:, :2] - asset.data.root_lin_vel_b[:, :2])
    scaled_error = error_sq[:, 0] / sigma_x + error_sq[:, 1] / sigma_y
    return torch.exp(-scaled_error)


def go2_track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Legacy Go2 yaw tracking reward with command- and terrain-dependent sigma."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    sigma = _go2_dynamic_sigma(
        env,
        torch.abs(command[:, 2]),
        v_min=1.0,
        v_max=2.0,
        default_sigma=0.25,
        max_sigma_by_terrain=_GO2_MAX_TRACKING_SIGMA_BY_TERRAIN,
    )

    error_sq = torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-error_sq / sigma)

# ===================== Extreme Parkour =====================
def extreme_parkour_tracking_goal_vel(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Reward world-frame xy velocity projected toward the current waypoint."""

    asset: Articulation = env.scene[asset_cfg.name]

    command_term = env.command_manager.get_term(command_name)

    target_pos_rel_w = (command_term.current_waypoint_w[:, :2] - asset.data.root_pos_w[:, :2]) # type:ignore

    target_norm = torch.linalg.vector_norm(target_pos_rel_w, dim=-1, keepdim=True,)

    target_direction_w = target_pos_rel_w / (target_norm + 1.0e-5)

    current_velocity_w = asset.data.root_lin_vel_w[:, :2]

    projected_velocity = torch.sum(target_direction_w * current_velocity_w, dim=-1)

    target_speed = command_term.forward_speed #type:ignore

    if target_speed.shape != (env.num_envs,):
       raise RuntimeError(
           "Extreme Parkour forward_speed must have shape "
           f"({env.num_envs},), got {tuple(target_speed.shape)}."
       )

    reward = torch.minimum(projected_velocity, target_speed,) / (target_speed + 1.0e-5)

    return reward

def extreme_parkour_tracking_yaw(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    command_term = env.command_manager.get_term(command_name)

    target_pos_rel_w = command_term.current_waypoint_w[:, :2] - asset.data.root_pos_w[:, :2] #type:ignore

    target_norm = torch.linalg.vector_norm(target_pos_rel_w, dim=-1, keepdim=True)

    target_direction_w = target_pos_rel_w / (target_norm + 1.0e-5)

    target_yaw_w = torch.atan2(target_direction_w[:, 1], target_direction_w[:, 0])

    robot_yaw_w = asset.data.heading_w

    yaw_error = target_yaw_w - robot_yaw_w

    return torch.exp(-torch.abs(yaw_error))

def extreme_parkour_lin_vel_z(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    command_term = env.command_manager.get_term(command_name)

    vertical_velocity_penalty = torch.square(asset.data.root_lin_vel_b[:, 2])

    is_flat =  command_term.terrain_class == command_term.cfg.parkour_flat_terrain_class # type: ignore

    penalty_scale = torch.where(is_flat, torch.ones_like(vertical_velocity_penalty), torch.full_like(vertical_velocity_penalty, 0.5),)

    return vertical_velocity_penalty * penalty_scale

def extreme_parkour_ang_vel_xy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=-1)

def extreme_parkour_orientation(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Penalize base tilt only on the local flat terrain class."""

    asset: Articulation = env.scene[asset_cfg.name]

    command_term = env.command_manager.get_term(command_name)

    orientation_penalty = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1)

    is_flat = command_term.terrain_class == command_term.cfg.parkour_flat_terrain_class # type:ignore

    return orientation_penalty * is_flat.float()

# 为什么需要 ManagerTermBase
# 普通函数只根据当前状态计算：f(state_t),但dof_acc 和 delta_torques 需要 f(state_t, state_t-1)
# ManagerTermBase 让奖励项拥有自己的持久状态，并允许 RewardManager 在 episode reset 时调用：term.reset(env_ids)
# 这可以保证不同 episode 之间不会继承上一回合的：
class ExtremeParkourJointAcceleration(ManagerTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: Articulation = env.scene[asset_cfg.name]

        if isinstance(asset_cfg.joint_ids, slice):
           raise ValueError("ExtremeParkourJointAcceleration requires 12 explicit joint IDs.")

        if len(asset_cfg.joint_ids) != 12:
            raise ValueError(
                "ExtremeParkourJointAcceleration expected "
                f"12 joints, got {len(asset_cfg.joint_ids)}."
            )

        self._asset = asset
        self._joint_ids = list(asset_cfg.joint_ids)

        self._last_joint_vel = torch.zeros(env.num_envs, 12, device=self.device)

    def reset(self, env_ids=None) -> None:
        # env_ids 指的是需要重置的并行环境编号
        # IsaacLab 的 RewardManager.reset() 内部会把 None 转换成：slice(None)
        if env_ids is None: # 手动/全局 reset：重置全部环境
            self._last_joint_vel.zero_()
        else:
            self._last_joint_vel[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        del asset_cfg

        current_joint_vel = self._asset.data.joint_vel[:, self._joint_ids]

        joint_acceleration = (self._last_joint_vel - current_joint_vel) / env.step_dt

        penalty = torch.sum(torch.square(joint_acceleration), dim=-1)

        self._last_joint_vel.copy_(current_joint_vel)

        return penalty

def extreme_parkour_action_rate(env: ManagerBasedRLEnv,) -> torch.Tensor:
    """Official L2 norm of the policy-step action difference."""

    current_action = env.action_manager.action
    previous_action = env.action_manager.prev_action

    if current_action.shape[-1] != 12:
        raise RuntimeError(
            f"Extreme Parkour expects a 12-D action, got shape {tuple(current_action.shape)}."
        )

    return torch.linalg.vector_norm(current_action - previous_action, dim=-1,)

class ExtremeParkourTorqueChange(ManagerTermBase):
    """Official policy-step torque-change penalty."""
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv,) -> None:
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: Articulation = env.scene[asset_cfg.name]

        if isinstance(asset_cfg.joint_ids, slice):
            raise ValueError("ExtremeParkourTorqueChange requires 12 explicit joint IDs.")

        if len(asset_cfg.joint_ids) != 12:
            raise ValueError(
                "ExtremeParkourTorqueChange expected "
                f"12 joints, got {len(asset_cfg.joint_ids)}."
            )

        self._asset = asset
        self._joint_ids = list(asset_cfg.joint_ids)

        self._last_torque = torch.zeros(env.num_envs, 12, device=env.device,)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._last_torque.zero_()
        else:
            self._last_torque[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,) -> torch.Tensor:
        del env, asset_cfg

        current_torque = self._asset.data.applied_torque[:, self._joint_ids,]

        penalty = torch.sum(torch.square(current_torque - self._last_torque), dim=-1,)

        self._last_torque.copy_(current_torque)

        return penalty

def extreme_parkour_collision(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 0.1,) -> torch.Tensor:
    """Count penalized bodies whose current contact force exceeds threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore

    current_forces_w = (contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]) # type:ignore

    force_magnitude = torch.linalg.vector_norm(current_forces_w, dim=-1,)

    collision = force_magnitude > threshold

    return torch.sum(collision.float(), dim=-1,)

def extreme_parkour_joint_position_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared joint-position deviation from the configured default pose."""

    asset: Articulation = env.scene[asset_cfg.name]

    joint_error = (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids])

    return torch.sum(torch.square(joint_error), dim=-1)

def extreme_parkour_feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, ratio: float = 4.0,) -> torch.Tensor:
    """Detect feet hitting near-vertical obstacle surfaces."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore

    current_forces_w = (contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]) # type: ignore

    horizontal_force = torch.linalg.vector_norm(current_forces_w[..., :2], dim=-1,)

    vertical_force = torch.abs(current_forces_w[..., 2])

    stumbled = torch.any(horizontal_force > ratio * vertical_force, dim=-1)

    return stumbled.float()

# 每只脚沿前进方向的射线位置为：x = -0.05, 0.00, +0.05 m
# 如果足端距离前后边缘不超过约 5 cm，相邻射线会出现：高度跳变超过 0.075 m,或者一条射线有命中，另一条没有命中。
# 0.075 m 来自mesh转换阈值：slope_threshold × horizontal_scale

class ExtremeParkourFeetEdge(ManagerTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]

        scanner_names: list[str] = cfg.params["scanner_names"] # type:ignore

        if isinstance(sensor_cfg.body_ids, slice):
           raise ValueError("ExtremeParkourFeetEdge requires four explicit foot body IDs.")

        if len(sensor_cfg.body_ids) != 4:
            raise ValueError(
                "ExtremeParkourFeetEdge expected four feet, "
                f"got {len(sensor_cfg.body_ids)}."
            )

        if len(scanner_names) != 4:
            raise ValueError(
                "ExtremeParkourFeetEdge expected four "
                f"foot scanners, got {len(scanner_names)}."
            )

        if len(set(scanner_names)) != 4:
            raise ValueError(
                "ExtremeParkourFeetEdge scanner names must be unique."
            )

        self._foot_body_ids = list(sensor_cfg.body_ids)

        self._scanners: list[RayCaster] = []

        for scanner_name in scanner_names:
            scanner: RayCaster = env.scene.sensors[scanner_name]  # type: ignore

            self._scanners.append(scanner)

        self._last_contact = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device,)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._last_contact.zero_()
        else:
            self._last_contact[env_ids] = False

    def __call__(self, env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        scanner_names: list[str],
        contact_threshold: float = 2.0,
        edge_height_threshold: float = 0.075,
        minimum_terrain_level: int = 4,
    ) -> torch.Tensor:
       del scanner_names

       contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type: ignore

       current_contact = torch.linalg.vector_norm(contact_sensor.data.net_forces_w[:, self._foot_body_ids,:], dim=-1) > contact_threshold # type:ignore

       filtered_contact = torch.logical_or(current_contact, self._last_contact)

       self._last_contact.copy_(current_contact)

       feet_at_edge = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device)

       for foot_index, scanner in enumerate(self._scanners):
           hit_height = scanner.data.ray_hits_w[..., 2].reshape(env.num_envs, 3, 3)

           valid_hit = torch.isfinite(hit_height)

           valid_x_pair = valid_hit[:, 1:, :] & valid_hit[:, :-1, :]

           height_jump = torch.abs(hit_height[:, 1:, :]- hit_height[:, :-1, :])

           edge_from_height = torch.any(valid_x_pair & ( height_jump > edge_height_threshold), dim=(1, 2),)

           # 场景：脚后跟踩在实地，脚前掌悬空在悬崖外
           validity_transition = valid_hit[:, 1:, :] ^ valid_hit[:, :-1, :]

           edge_from_missing_hit = torch.any(validity_transition, dim=(1, 2))

           # 假设机器人脚掌完全悬空（正在空中），此时脚后跟处的射线可能扫到远处的墙壁（有效），
           # 脚前掌射向深渊（无效）。如果只看判据二，会误以为“踩在边缘”。但此时脚根本没落地（中心点无效）， 不应该触发“踩边缘”惩罚
           center_hit_is_valid = valid_hit[:, 1, 1]

           feet_at_edge[:, foot_index] = center_hit_is_valid & (edge_from_height | edge_from_missing_hit)

       terrain_levels = env.scene.terrain.terrain_levels # type:ignore

       # 在平地上（等级 0），没有台阶边缘，检测没意义。只有当地形复杂到一定级别（>=4）时，机器人才需要刻意避开边缘
       difficult_enough = terrain_levels >= minimum_terrain_level

       contacting_edge = filtered_contact & feet_at_edge

       return difficult_enough.float() * torch.sum(contacting_edge.float(), dim=-1)
