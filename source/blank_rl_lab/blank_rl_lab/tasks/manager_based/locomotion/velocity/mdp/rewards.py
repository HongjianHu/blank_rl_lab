from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
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