from __future__ import annotations

import torch
from typing import TYPE_CHECKING, cast

from isaaclab.managers import SceneEntityCfg, ActionManager, ManagerTermBase, RewardTermCfg as RewTerm
from isaaclab.assets import Articulation
from isaaclab.sensors import RayCaster, ContactSensor
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import joint_deviation_l1
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


# =============================================================================
# Velocity tracking rewards (override IsaacLab with gravity gating + asset_cfg)
# =============================================================================

def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel.

    This is an override of the IsaacLab built-in ``track_lin_vel_xy_exp`` with two
    enhancements:

    1. **Gravity gating**: The reward is scaled by
       ``clamp(-projected_gravity_b[:, 2], 0, 0.7) / 0.7``, which smoothly
       decays to zero when the robot tilts beyond ~45°.  This prevents the
       policy from being penalised for failing to track velocity while falling
       over.

    2. **Explicit ``asset_cfg``**: Allows callers to specify which rigid body
       provides the velocity measurement.

    Args:
        env: The RL environment instance.
        std: Standard deviation of the exponential kernel (smaller → tighter).
        command_name: Name of the velocity command term (e.g. ``"base_velocity"``).
        asset_cfg: Scene entity configuration for the tracked body.

    Returns:
        Reward tensor of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    # Gravity gating: decay reward when robot is tilted / falling
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel.

    Override of IsaacLab built-in with gravity gating and explicit ``asset_cfg``.
    See :func:`track_lin_vel_xy_exp` for the rationale of the gravity gating.

    Args:
        env: The RL environment instance.
        std: Standard deviation of the exponential kernel.
        command_name: Name of the velocity command term.
        asset_cfg: Scene entity configuration for the tracked body.

    Returns:
        Reward tensor of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2]
    )
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands in the **gravity-aligned yaw frame**.

    Unlike :func:`track_lin_vel_xy_exp` which uses ``root_lin_vel_b`` (body frame),
    this function rotates the world-frame linear velocity by the yaw component of
    the base orientation.  This removes the influence of roll/pitch on the velocity
    error, which is especially useful on rough terrain where the base constantly
    tilts.

    The reward is computed as::

        vel_yaw = quat_apply_inverse(yaw_quat(root_quat_w), root_lin_vel_w[:, :3])
        error = (cmd_xy - vel_yaw_xy)^2
        reward = exp(-error / std^2) * gravity_gating

    Args:
        env: The RL environment instance.
        std: Standard deviation of the exponential kernel.
        command_name: Name of the velocity command term.
        asset_cfg: Scene entity configuration.

    Returns:
        Reward tensor of shape ``(num_envs,)``.
    """
    asset = env.scene[asset_cfg.name]
    # Project world-frame velocity into the yaw-aligned frame (drop roll/pitch)
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in the **world frame**.

    Uses ``root_ang_vel_w`` instead of ``root_ang_vel_b``, so the yaw-rate
    tracking is evaluated in the world z-axis regardless of the robot's
    orientation.

    Args:
        env: The RL environment instance.
        command_name: Name of the velocity command term.
        std: Standard deviation of the exponential kernel.
        asset_cfg: Scene entity configuration.

    Returns:
        Reward tensor of shape ``(num_envs,)``.
    """
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2]
    )
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# =============================================================================
# Joint power & penalisation (Go2W-specific additions)
# =============================================================================

def joint_power(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise mechanical joint power :math:`P = |\\tau \\cdot \\omega|`.

    This is a more physically meaningful energy penalty than ``joint_torques_l2``
    alone, because a large torque consumes little power if the joint is not
    moving, whereas a small torque at high speed can still waste energy.

    Computed as::

        reward = sum(|joint_vel * applied_torque|, dim=1)

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity configuration specifying which joints to include.

    Returns:
        Penalty tensor of shape ``(num_envs,)`` (higher value = more power).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(
        torch.abs(
            asset.data.joint_vel[:, asset_cfg.joint_ids]
            * asset.data.applied_torque[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    return reward


# =============================================================================
# Standing / moving behaviour shaping
# =============================================================================

def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise joint deviations from default pose **only when the robot is idle**.

    When the velocity command magnitude is below ``command_threshold``, the
    robot is expected to stand still in its default joint configuration.  This
    term computes ``joint_deviation_l1`` and gates it by the command magnitude.

    .. math::

        reward = joint\\_deviation\\_l1 \\times
                 \\mathbb{1}[\\|cmd\\| < command\\_threshold] \\times
                 gravity\\_gating

    Args:
        env: The RL environment instance.
        command_name: Name of the velocity command term.
        command_threshold: L2-norm of command below which the robot is
            considered idle.
        asset_cfg: Scene entity configuration for the joints.

    Returns:
        Penalty tensor of shape ``(num_envs,)``.
    """
    reward = joint_deviation_l1(env, asset_cfg)
    # Gate: only apply when the robot is NOT commanded to move
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalise joint position error from default, with **escalated penalty when idle**.

    This is a composite penalty that distinguishes two regimes:

    * **Moving** (cmd > threshold or body_vel > threshold):
      Standard L2 distance from default pose.
    * **Standing** (cmd ≤ threshold and body_vel ≤ threshold):
      Penalty multiplied by ``stand_still_scale`` (e.g. 5×).

    This prevents the robot from adopting awkward poses while stationary,
    while still allowing postural deviations during locomotion.

    .. math::

        running\\_error = \\|joint\\_pos - default\\_joint\\_pos\\|_2

        reward = \\begin{cases}
            running\\_error               & \\text{if moving} \\\\
            stand\\_still\\_scale \\times running\\_error & \\text{if standing}
        \\end{cases}

    Args:
        env: The RL environment instance.
        command_name: Name of the velocity command term.
        asset_cfg: Scene entity configuration.
        stand_still_scale: Multiplier applied when the robot is stationary.
        velocity_threshold: Body velocity (m/s) above which robot is considered moving.
        command_threshold: Command magnitude above which robot is considered moving.

    Returns:
        Penalty tensor of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)

    running_reward = torch.linalg.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids],
        dim=1,
    )

    # Escalate penalty when standing still
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise wheel joint velocity, with different behaviour for moving vs standing.

    This is a **Go2W-specific** penalty that prevents the wheels from spinning
    uselessly:

    * **Moving** (cmd > threshold or body_vel > threshold):
      Only penalises wheels that are **in the air** (no ground contact), because
      spinning a wheel that is not touching the ground wastes energy and produces
      destabilising angular momentum.
    * **Standing** (otherwise):
      Penalises **all** wheel velocity, encouraging the wheels to stay still
      when the robot is not commanded to move.

    .. math::

        running  = \\sum (in\\_air \\times |wheel\\_vel|)
        standing = \\sum |wheel\\_vel|

    Args:
        env: The RL environment instance.
        sensor_cfg: Contact sensor configuration for detecting wheel-ground contact.
        command_name: Name of the velocity command term.
        velocity_threshold: Body velocity threshold for moving detection.
        command_threshold: Command threshold for moving detection.
        asset_cfg: Scene entity configuration for the joints (typically wheels).

    Returns:
        Penalty tensor of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])

    # Determine which wheels are in the air
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type:ignore
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]

    running_reward = torch.sum(in_air * joint_vel, dim=1)     # only penalise airborne wheels
    standing_reward = torch.sum(joint_vel, dim=1)             # penalise all wheels

    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


# =============================================================================
# Symmetry / synchronisation rewards
# =============================================================================

def joint_mirror(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]],
) -> torch.Tensor:
    """Penalise position asymmetry between mirrored joint pairs.

    For a quadruped with a trot gait, the diagonal leg pairs should move in
    sync.  This term compares ``FR ↔ RL`` and ``FL ↔ RR`` and penalises the
    squared difference in joint positions.

    .. math::

        reward = \\frac{1}{N_{pairs}} \\sum_{(a,b) \\in pairs}
                 \\|joint\\_pos[a] - joint\\_pos[b]\\|^2

    Joint indices are cached on ``env`` to avoid recomputation.

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity configuration.
        mirror_joints: List of joint-name pairs to compare, e.g.
            ``[["FR.*", "RL.*"], ["FL.*", "RR.*"]]``.

    Returns:
        Penalty tensor of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None: # type:ignore
        env.joint_mirror_joints_cache = [               # type:ignore
            [asset.find_joints(joint_name) for joint_name in joint_pair]
            for joint_pair in mirror_joints
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_pair in env.joint_mirror_joints_cache:    # type:ignore
        diff = torch.sum(
            torch.square(
                asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]
            ),
            dim=-1,
        )
        reward += diff

    reward *= 1.0 / len(mirror_joints) if len(mirror_joints) > 0 else 0.0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]],
) -> torch.Tensor:
    """Penalise action asymmetry between mirrored joint pairs.

    Similar to :func:`joint_mirror`, but compares the **absolute values** of
    the policy's action outputs rather than actual joint positions.  Using
    absolute values allows the two legs in a pair to have opposite signs
    (e.g. one swinging forward, the other backward) while still requiring
    equal magnitude.

    .. math::

        reward = \\frac{1}{N_{pairs}} \\sum_{(a,b) \\in pairs}
                 (|action[a]| - |action[b]|)^2

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity configuration.
        mirror_joints: List of joint-name pairs, e.g.
            ``[["FR.*", "RL.*"], ["FL.*", "RR.*"]]``.

    Returns:
        Penalty tensor of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None: # type:ignore
        env.action_mirror_joints_cache = [  # type:ignore
            [asset.find_joints(joint_name) for joint_name in joint_pair]
            for joint_pair in mirror_joints
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_pair in env.action_mirror_joints_cache: # type:ignore
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff

    reward *= 1.0 / len(mirror_joints) if len(mirror_joints) > 0 else 0.0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    joint_groups: list[list[str]],
) -> torch.Tensor:
    """Penalise action variance within groups of same-type joints.

    For a quadruped, the four hip joints (FR/FL/RL/RR) should ideally move
    with similar magnitudes to produce a coordinated gait.  This term computes
    the **variance** of the absolute action values within each group and
    penalises it.

    .. math::

        reward = \\frac{1}{N_{groups}} \\sum_{g \\in groups}
                 \\text{Var}(|actions[g]|)

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity configuration.
        joint_groups: List of joint-name groups, e.g.::

            [
                ["FR_hip_joint", "FL_hip_joint", "RL_hip_joint", "RR_hip_joint"],
                ["FR_thigh_joint", "FL_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
                ["FR_calf_joint", "FL_calf_joint", "RL_calf_joint", "RR_calf_joint"],
            ]

    Returns:
        Penalty tensor of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None: # type:ignore
        env.action_sync_joint_cache = [ # type:ignore
            [asset.find_joints(joint_name) for joint_name in joint_group]
            for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_group in env.action_sync_joint_cache: # type:ignore
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compute variance

        # Stack absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group],
            dim=1,
        )  # (num_envs, num_joints_in_group)

        mean_actions = torch.mean(actions, dim=1, keepdim=True)
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)
        reward += variance.squeeze()

    reward *= 1.0 / len(joint_groups) if len(joint_groups) > 0 else 0.0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# =============================================================================
# Gait enforcement (trot gait reward via contact timing)
# =============================================================================

class GaitReward(ManagerTermBase):
    """Gait-enforcing reward that biases the policy towards a **trot gait**.

    A trot gait for a quadruped requires two diagonal foot pairs to move in
    synchrony: (FR, RL) as one pair and (FL, RR) as the other.  This reward
    penalises **contact-timing mismatches** between the paired feet.

    **How it works**:

    1. **Sync reward** (×2): For each synchronised pair, the air-time and
       contact-time values of the two feet should match.
    2. **Async reward** (×4): For each cross-pair combination, the air-time of
       one foot should equal the contact-time of the other (anti-phase).

    The final reward is the product of all six terms, so **all gait conditions
    must be simultaneously satisfied** to achieve a high reward.

    The reward is gated to zero when the command is small and the robot is
    stationary, so idle standing is not penalised for having a non-trot contact
    pattern.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialise the gait reward term.

        Args:
            cfg: Reward term configuration.  Expected ``params`` keys:

                * ``std``: Kernel standard deviation.
                * ``command_name``: Velocity command name.
                * ``max_err``: Maximum squared error (for clipping).
                * ``velocity_threshold``: Body velocity threshold for activation.
                * ``command_threshold``: Command threshold for activation.
                * ``synced_feet_pair_names``: Tuple of two 2-tuples of foot body
                  name regex patterns, e.g.
                  ``(("FR.*_foot", "RL.*_foot"), ("FL.*_foot", "RR.*_foot"))``.
                * ``sensor_cfg``: Contact sensor configuration.
                * ``asset_cfg``: Articulation asset configuration.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"] # type:ignore
        self.command_name: str = cfg.params["command_name"] # type:ignore
        self.max_err: float = cfg.params["max_err"] # type:ignore
        self.velocity_threshold: float = cfg.params["velocity_threshold"] # type:ignore
        self.command_threshold: float = cfg.params["command_threshold"] # type:ignore
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name] # type:ignore
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]

        # Resolve foot body names to body indices
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2       # type:ignore
            or len(synced_feet_pair_names[0]) != 2 # type:ignore
            or len(synced_feet_pair_names[1]) != 2 # type:ignore
        ):
            raise ValueError(
                "GaitReward only supports gaits with exactly two pairs of "
                "synchronised feet (e.g. trot)."
            )
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]  # type:ignore
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]  # type:ignore
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the trot-gait reward.

        Args:
            env: The RL environment instance.

        Returns:
            Reward tensor of shape ``(num_envs,)``.  Zero when the robot is
            stationary (cmd ≈ 0 and body_vel ≈ 0).
        """
        # Sync: paired feet should have matching air / contact times
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1

        # Async: cross-pair feet should be in opposite phase
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3

        # Gate: only enforce gait when the robot is moving
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            torch.tensor(0.0, device=env.device),
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronisation of two feet (air/contact times should match)."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time

        se_air = torch.clip(
            torch.square(air_time[:, foot_0] - air_time[:, foot_1]), # type:ignore
            max=self.max_err**2,
        )
        se_contact = torch.clip(
            torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), # type:ignore
            max=self.max_err**2,
        )
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronisation of two feet (one in air ↔ other in contact)."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time

        se_act_0 = torch.clip(
            torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), # type:ignore
            max=self.max_err**2,
        )
        se_act_1 = torch.clip(
            torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), # type:ignore
            max=self.max_err**2,
        )
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


# =============================================================================
# Feet / contact rewards
# =============================================================================

def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    Override of IsaacLab built-in with gravity gating.  Rewards the agent for
    taking steps where the foot spends more than ``threshold`` seconds in the
    air.  The reward is zero when the velocity command is near zero.

    .. math::

        reward = \\sum_{feet} (last\\_air\\_time - threshold) \\times
                 first\\_contact \\times
                 \\mathbb{1}[\\|cmd\\| > 0.1] \\times
                 gravity\\_gating

    where ``first_contact`` is 1 at the instant the foot touches down (so the
    reward fires once per step).

    Args:
        env: The RL environment instance.
        command_name: Name of the velocity command term.
        sensor_cfg: Contact sensor configuration for the feet.
        threshold: Minimum air-time (seconds) to be rewarded.

    Returns:
        Reward tensor of shape ``(num_envs,)``.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type:ignore
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids] # type:ignore
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # No reward when not commanded to move
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward long steps for **biped** robots with single-stance constraint.

    Unlike the quadruped version, this rewards **single-foot stance** (exactly
    one foot in contact) and caps the mode time at ``threshold``.

    Note:
        This is provided for completeness but is **not** typically used for a
        Go2W quadruped.

    Args:
        env: The RL environment instance.
        command_name: Velocity command name.
        threshold: Max rewarded stance time.
        sensor_cfg: Contact sensor configuration.

    Returns:
        Reward tensor of shape ``(num_envs,)``.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type:ignore
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids] # type:ignore
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] # type:ignore
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalise variance in air-time / contact-time across feet.

    A large variance means some feet spend much more (or less) time in the air
    than others — an asymmetric, inefficient gait.  This term encourages
    symmetric foot use.

    .. math::

        reward = \\text{Var}(clip(air\\_time, max=0.5))
               + \\text{Var}(clip(contact\\_time, max=0.5))

    Args:
        env: The RL environment instance.
        sensor_cfg: Contact sensor configuration.

    Returns:
        Penalty tensor of shape ``(num_envs,)``.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type:ignore
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids] # type:ignore
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids] # type:ignore
    reward = (
        torch.var(torch.clip(last_air_time, max=0.5), dim=1)
        + torch.var(torch.clip(last_contact_time, max=0.5), dim=1)
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    expect_contact_num: int,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward maintaining exactly ``expect_contact_num`` feet in contact while moving.

    For a trot gait, exactly **2** feet should be on the ground at all times
    during locomotion.  This term penalises any deviation (0, 1, 3, or 4 feet
    in contact) when the robot is commanded to move.

    .. math::

        reward = (num\\_in\\_contact \\neq expect\\_contact\\_num).float()
               \\times \\mathbb{1}[\\|cmd\\| > 0.1]
               \\times gravity\\_gating

    Args:
        env: The RL environment instance.
        command_name: Velocity command name.
        expect_contact_num: Expected number of feet in contact (2 for trot).
        sensor_cfg: Contact sensor configuration.

    Returns:
        Penalty tensor of shape ``(num_envs,)``.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type:ignore
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward **all feet on the ground** when the robot is idle.

    Complementary to :func:`feet_contact`: when there is **no** velocity
    command, the robot should stand with all four feet firmly planted.

    .. math::

        reward = (num\\_in\\_contact) \\times
                 \\mathbb{1}[\\|cmd\\| < 0.1] \\times
                 gravity\\_gating

    Args:
        env: The RL environment instance.
        command_name: Velocity command name.
        sensor_cfg: Contact sensor configuration.

    Returns:
        Reward tensor of shape ``(num_envs,)`` (higher = more feet in contact).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type:ignore
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    # Only active when NOT commanded to move
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise feet **sliding** laterally while in contact with the ground.

    When a foot is on the ground, the lateral velocity (xy) in the **body
    frame** should be near zero.  This term penalises the norm of the foot's
    lateral velocity multiplied by a binary contact indicator.

    .. math::

        reward = \\sum_{feet} (contact \\times \\|foot\\_vel\\_body\\_xy\\|)

    where ``contact`` is determined from force sensor history (max force > 1 N).

    Args:
        env: The RL environment instance.
        sensor_cfg: Contact sensor configuration.
        asset_cfg: Scene entity configuration for the foot bodies.

    Returns:
        Penalty tensor of shape ``(num_envs,)``.
    """
    # Binary contact: foot has experienced > 1 N force recently
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type:ignore
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :] # type:ignore
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )
    asset: Articulation = env.scene[asset_cfg.name]

    # Compute foot velocity in body frame (subtract root velocity)
    cur_footvel_translated = (
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
        - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    )
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device) # type:ignore
    for i in range(len(asset_cfg.body_ids)): # type:ignore
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_lateral_vel = torch.sqrt(
        torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)
    ).view(env.num_envs, -1)

    reward = torch.sum(foot_lateral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward swinging feet for clearing a specified height above the ground.

    This encourages the robot to lift its feet sufficiently during the swing
    phase.  The penalty is gated by the foot's horizontal velocity via a
    ``tanh``, so **only actively swinging feet** contribute.

    .. math::

        error = (foot\\_z - target\\_height)^2
        gate  = \\tanh(tanh\\_mult \\times \\|foot\\_vel\\_xy\\|)
        reward = \\sum (error \\times gate)

    Args:
        env: The RL environment instance.
        command_name: Velocity command name (rewards only when cmd > 0.1).
        asset_cfg: Scene entity configuration for the foot bodies (world frame).
        target_height: Desired foot clearance height in **world** coordinates.
        tanh_mult: Multiplier for the velocity gate sharpness.

    Returns:
        Penalty tensor of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height
    )
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward swinging feet for clearing a height relative to the **body frame**.

    Similar to :func:`feet_height`, but foot positions and velocities are
    expressed in the body frame by subtracting the root pose and applying the
    inverse quaternion.  This makes the reward independent of the robot's
    absolute height (e.g. when jumping or on slopes).

    The ``target_height`` is typically negative (e.g. -0.3 m) meaning the foot
    should be *below* the body by that amount.

    Args:
        env: The RL environment instance.
        command_name: Velocity command name.
        asset_cfg: Scene entity configuration for the foot bodies.
        target_height: Desired foot height in body-frame z (negative = below body).
        tanh_mult: Multiplier for the velocity gate sharpness.

    Returns:
        Penalty tensor of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Transform foot positions to body frame
    cur_footpos_translated = (
        asset.data.body_pos_w[:, asset_cfg.body_ids, :]
        - asset.data.root_pos_w[:, :].unsqueeze(1)
    )
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device) # type:ignore
    for i in range(len(asset_cfg.body_ids)): # type:ignore
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )

    # Transform foot velocities to body frame
    cur_footvel_translated = (
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
        - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    )
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device) # type:ignore
    for i in range(len(asset_cfg.body_ids)): # type:ignore
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )

    foot_z_target_error = torch.square(
        footpos_in_body_frame[:, :, 2] - target_height
    ).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_y_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward maintaining the desired lateral stance width.

    Each foot should be approximately ``±stance_width/2`` from the body centre
    in the y-direction.  The reward uses an exponential kernel on the squared
    deviation.

    .. math::

        desired\\_y_i = \\pm stance\\_width \\,/\\, 2
        reward = \\exp\\!\\left(-\\frac{\\sum_i (y_i - desired\\_y_i)^2}{std^2}\\right)

    Args:
        env: The RL environment instance.
        stance_width: Total desired lateral distance between left and right feet.
        std: Kernel standard deviation.
        asset_cfg: Scene entity configuration for the foot bodies.

    Returns:
        Reward tensor of shape ``(num_envs,)`` (1 = perfect stance width).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cur_footsteps_translated = (
        asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
        - asset.data.root_link_pos_w[:, :].unsqueeze(1)
    )

    n_feet = len(asset_cfg.body_ids) # type:ignore
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w),
            cur_footsteps_translated[:, i, :],
        )

    # Alternate sign: left feet +, right feet - (or vice versa)
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2.0 * side_sign.unsqueeze(0)

    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward maintaining desired stance **width and length** simultaneously.

    Like :func:`feet_distance_y_exp` but also constrains the x (longitudinal)
    foot positions for a rectangular stance.  Uses the same exponential kernel.

    .. math::

        desired\\_x = [L/2, L/2, -L/2, -L/2]  \\quad (FR, FL, RR, RL)
        desired\\_y = [W/2, -W/2, W/2, -W/2]

    Args:
        env: The RL environment instance.
        stance_width: Desired lateral distance between left and right feet.
        stance_length: Desired longitudinal distance between front and rear feet.
        std: Kernel standard deviation.
        asset_cfg: Scene entity configuration for the foot bodies (must have 4).

    Returns:
        Reward tensor of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cur_footsteps_translated = (
        asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
        - asset.data.root_link_pos_w[:, :].unsqueeze(1)
    )

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w),
            cur_footsteps_translated[:, i, :],
        )

    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    # FR, FL positive x (front); RR, RL negative x (rear)
    desired_xs = torch.cat(
        [
            stance_length_tensor / 2,
            stance_length_tensor / 2,
            -stance_length_tensor / 2,
            -stance_length_tensor / 2,
        ],
        dim=1,
    )
    # FR, RR positive y; FL, RL negative y (right / left)
    desired_ys = torch.cat(
        [
            stance_width_tensor / 2,
            -stance_width_tensor / 2,
            stance_width_tensor / 2,
            -stance_width_tensor / 2,
        ],
        dim=1,
    )

    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# =============================================================================
# Override base penalties (gravity gating + asset_cfg)
# =============================================================================

def upward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the robot for keeping its torso upright.

    Penalises deviation of the projected gravity vector from the body z-axis::

        reward = (1 - projected_gravity_b[:, 2])²

    When the robot is perfectly upright, ``projected_gravity_b[:, 2]`` ≈ 1
    and the reward approaches 0 (no penalty).  As the robot tilts, the value
    grows.

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity configuration.

    Returns:
        Penalty tensor of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.square(1.0 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalise base height deviation from target.

    Override of IsaacLab built-in with two enhancements:

    1. **Rough terrain support**: When ``sensor_cfg`` is provided, ray-cast
       sensor readings are used to compute an adjusted target height relative
       to the local ground surface.
    2. **Gravity gating**: Reward decays when the robot tilts.

    .. math::

        adjusted\\_target = target\\_height + mean(ray\\_hits\\_z)
        reward = (base\\_z - adjusted\\_target)^2 \\times gravity\\_gating

    Args:
        env: The RL environment instance.
        target_height: Nominal target height in world frame.
        asset_cfg: Scene entity configuration for the base body.
        sensor_cfg: Optional ray-caster configuration for terrain height sensing.

    Returns:
        Penalty tensor of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        # Guard against invalid / extreme ray values
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        adjusted_target_height = target_height

    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise z-axis base linear velocity using L2 squared kernel.

    Override of IsaacLab built-in with gravity gating and explicit ``asset_cfg``.

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity configuration.

    Returns:
        Penalty tensor of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise xy-axis base angular velocity using L2 squared kernel.

    Override of IsaacLab built-in with gravity gating and explicit ``asset_cfg``.

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity configuration.

    Returns:
        Penalty tensor of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalise undesired body contacts.

    Override of IsaacLab built-in with gravity gating.  Counts the number of
    body parts whose contact force exceeds ``threshold``.

    Args:
        env: The RL environment instance.
        threshold: Force threshold (N) above which a contact is flagged.
        sensor_cfg: Contact sensor configuration.

    Returns:
        Penalty tensor of shape ``(num_envs,)`` (integer count of contacts).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type:ignore
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = (
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] # type:ignore
        > threshold
    )
    reward = torch.sum(is_contact, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise non-flat base orientation using L2 squared kernel.

    Override of IsaacLab built-in with gravity gating and explicit ``asset_cfg``.
    Penalises the xy-components of the projected gravity vector; when the robot
    is perfectly flat, these are zero.

    Args:
        env: The RL environment instance.
        asset_cfg: Scene entity configuration.

    Returns:
        Penalty tensor of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

class TerrainAwareSwingFootClearanceReward(ManagerTermBase):
    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        contact_sensor_cfg: SceneEntityCfg = cfg.params["contact_sensor_cfg"]
        height_sensor_cfg: SceneEntityCfg = cfg.params["height_sensor_cfg"]

        self._asset: Articulation = env.scene[asset_cfg.name]
        self._contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name] # type:ignore
        self._height_sensor: RayCaster = env.scene.sensors[height_sensor_cfg.name] # type:ignore
        # SceneEntityCfg 会在 Manager 初始化时把名字解析为索引；名字解析是相对昂贵的操作，应只执行一次
        self._asset_foot_ids = self._require_explicit_indices(asset_cfg.body_ids, name="asset.body_ids")
        self._contact_foot_ids = self._require_explicit_indices(contact_sensor_cfg.body_ids, name="contact_sensor_cfg.body_ids")
        if len(self._asset_foot_ids) == 0: # type:ignore
            raise ValueError(
                "TerrainAwareSwingFootClearanceReward 至少需要一个足端。"
            )

        if len(self._asset_foot_ids) != len(self._contact_foot_ids): # type:ignore
            raise ValueError(
                "机器人足端数量和接触传感器足端数量不一致："
                f"{len(self._asset_foot_ids)} != "  # type:ignore
                f"{len(self._contact_foot_ids)}"    # type:ignore
            )        

    asset_foot_names = [self._asset.body_names[index] for index in self._asset_foot_ids] # type:ignore
    contact_foot_names = [self._contact_sensor.body_names[index] for index in self._contact_foot_ids] # type:ignore
    if asset_foot_names != contact_foot_names:
        raise ValueError(
            "机器人足端和接触传感器足端的顺序不一致。\n"
            f"Articulation 顺序: {asset_foot_names}\n"
            f"ContactSensor 顺序: {contact_foot_names}\n"
            "请对两个 SceneEntityCfg 都设置相同的显式 body_names,"
            "并设置 preserve_order=True。"
        )
    
    self._foot_names = tuple(asset_foot_names) # type:ignore
    self._num_feet = len(self._foot_names) # type:ignore

    if not self._contact_sensor.cfg.track_air_time:  # type:ignore
        raise ValueError(
            f"ContactSensor '{contact_sensor_cfg.name}' "  # type:ignore
            "必须设置 track_air_time=True,"
            "否则 current_air_time 不可用。"
        )

    self._k_nearest_rays = int(cfg.params["k_nearest_rays"]) # type:ignore

    if self._k_nearest_rays <= 0: # type:ignore
        raise ValueError(
            "k_nearest_rays 必须为正整数。"
        )
    
    self._runtime_shape_checked = False # type:ignore



    

    @staticmethod
    def _require_explicit_indices(indices: list[int] | slice, name: str):
        if isinstance(indices, slice):
            raise ValueError(
                f"{name} 当前为 slice(None)。"
                "这个奖励要求显式给出 body_names，"
                "不能默认选择所有刚体。"
            )
        return list[indices]