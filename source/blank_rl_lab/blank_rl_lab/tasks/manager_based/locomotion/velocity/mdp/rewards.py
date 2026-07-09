from __future__ import annotations

import torch
from typing import TYPE_CHECKING, cast

from isaaclab.managers import SceneEntityCfg, ActionManager
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
