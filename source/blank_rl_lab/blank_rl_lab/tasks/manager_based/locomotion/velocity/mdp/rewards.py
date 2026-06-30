from __future__ import annotations

import torch
from typing import TYPE_CHECKING, cast

from isaaclab.managers import SceneEntityCfg, ActionManager
from isaaclab.assets import Articulation

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

    # 计算二阶差分
    second_order_diff = action_t - 2.0 * action_t1 + action_t2  # (num_envs, action_dim)

    # 更新 buffer：将 a_{t-1} 存为下一步的 a_{t-2}
    setattr(env, "_action_t2", action_t1.clone())

    # 返回每个 env 的惩罚标量（对 action_dim 求和）
    penalty = torch.sum(second_order_diff ** 2, dim=-1)  # shape: (num_envs,)

    return penalty