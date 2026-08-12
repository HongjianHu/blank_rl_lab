from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Any
from dataclasses import MISSING
from isaaclab.envs import ManagerBasedEnv
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera, ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from blank_rl_lab.envs import ManagerBasedAnimationEnv
    from blank_rl_lab.managers import AnimationTerm


def root_rot_tan_norm(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    root_quat = robot.data.root_quat_w
    root_rotm = math_utils.matrix_from_quat(root_quat)

    # use the first and last column of the rotation matrix as the tangent and normal vectors
    tan_vec = root_rotm[:, :, 0]  # (N, 3)
    norm_vec = root_rotm[:, :, 2]  # (N, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (N, 6)

    return obs


def key_body_pos_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=MISSING, preserve_order=True), # type:ignore
) -> torch.Tensor:

    robot: Articulation = env.scene[asset_cfg.name]

    key_body_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]  # shape: (num_envs, M, 3)
    root_pos_w = robot.data.root_pos_w      # shape: (num_envs, 3).
    root_quat = robot.data.root_quat_w    # shape: (num_envs, 4), w, x, y, z order.

    num_key_bodies = key_body_pos_w.shape[1]
    num_envs = root_pos_w.shape[0]

    key_body_pos_b = math_utils.quat_apply_inverse(
        root_quat.unsqueeze(1).expand(-1, num_key_bodies, -1),
        key_body_pos_w - root_pos_w.unsqueeze(1).expand(-1, num_key_bodies, -1)
    )

    return key_body_pos_b.reshape(num_envs, -1)


def root_height(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root height relative to the environment origin, matching Go2 AMP ``z_pos``."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]


def amp_observation(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=MISSING, preserve_order=True), # type:ignore
    include_key_body_pos_b: bool = True,
    include_base_motion: bool = True,
    include_root_height: bool = True,
) -> torch.Tensor:
    """Current Go2 AMP observation.

    The full 43-D layout matches ``go2_amp.py::get_amp_observations()``:
    joint_pos(12), foot_pos_b(12), base_lin_vel_b(3), base_ang_vel_b(3),
    joint_vel(12), root_z(1).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    obs_terms = [robot.data.joint_pos]
    if include_key_body_pos_b:
        obs_terms.append(key_body_pos_b(env, asset_cfg))
    if include_base_motion:
        obs_terms.extend([robot.data.root_lin_vel_b, robot.data.root_ang_vel_b])
    obs_terms.append(robot.data.joint_vel)
    if include_root_height:
        obs_terms.append(root_height(env, SceneEntityCfg(asset_cfg.name)))
    return torch.cat(obs_terms, dim=-1)


def ref_root_pos_error(
    env: ManagerBasedAnimationEnv,
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    abs_height: bool = True
) -> torch.Tensor:
    """Compute the difference between robot root position and reference motion root position.

    The function calculates: reference_root_pos - current_robot_root_pos

    Args:
        env: The animation environment.
        animation: Name of the animation term to use as reference.
        asset_cfg: Configuration for the robot asset.
        abs_height: If True, use absolute height from reference motion (returns 3D position).
                   If False, only return horizontal displacement (2D: x, y only).

    Returns:
        Flattened tensor with shape:
        - (num_envs, num_steps * 3) if abs_height=True
        - (num_envs, num_steps * 2) if abs_height=False

    Note:
        Positive values indicate the reference motion is ahead/above the robot.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    ref_root_pos_w = animation_term.get_root_pos_w()  # shape: (num_envs, num_steps, 3)
    root_pos_w = robot.data.root_pos_w - env.scene.env_origins  # shape: (num_envs, 3)

    num_envs = root_pos_w.shape[0]

    # Compute position difference: ref - current
    # Broadcasting handles the dimension expansion automatically
    pos_diff = ref_root_pos_w - root_pos_w.unsqueeze(1)  # shape: (num_envs, num_steps, 3)

    if abs_height:
        # Replace relative z with absolute reference height
        pos_diff[:, :, 2] = ref_root_pos_w[:, :, 2]
        return pos_diff.reshape(num_envs, -1)  # shape: (num_envs, num_steps * 3)
    else:
        # Only return horizontal displacement (x, y)
        return pos_diff[:, :, :2].reshape(num_envs, -1)  # shape: (num_envs, num_steps * 2)


def ref_root_rot_tan_norm(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:

    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    ref_root_quat = animation_term.get_root_quat()  # shape: (num_envs, num_steps, 4)
    ref_root_rotm = math_utils.matrix_from_quat(ref_root_quat)  # shape: (num_envs, num_steps, 3, 3)
    ref_root_tan_vec = ref_root_rotm[:, :, :, 0]  # shape: (num_envs, num_steps, 3)
    ref_root_norm_vec = ref_root_rotm[:, :, :, 2]  # shape: (num_envs, num_steps, 3)
    obs = torch.cat([ref_root_tan_vec, ref_root_norm_vec], dim=-1)  # shape: (num_envs, num_steps, 6)

    if flatten_steps_dim:
        return obs.reshape(env.num_envs, -1)
    else:
        return obs


def ref_root_ang_vel_b(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:

    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs

    ref_root_ang_vel_w = animation_term.get_root_ang_vel_w()  # shape: (num_envs, num_steps, 3)
    ref_root_quat = animation_term.get_root_quat()  # shape: (num_envs, num_steps, 4)
    ref_root_ang_vel = math_utils.quat_apply_inverse(
        ref_root_quat, ref_root_ang_vel_w
    )

    if flatten_steps_dim:
        return ref_root_ang_vel.reshape(num_envs, -1)
    else:
        return ref_root_ang_vel


def ref_root_lin_vel_b(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:

    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs

    ref_root_vel_w = animation_term.get_root_vel_w()
    ref_root_quat = animation_term.get_root_quat()
    ref_root_vel_b = math_utils.quat_apply_inverse(ref_root_quat, ref_root_vel_w)

    if flatten_steps_dim:
        return ref_root_vel_b.reshape(num_envs, -1)
    else:
        return ref_root_vel_b


def ref_joint_pos(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:

    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    ref_dof_pos = animation_term.get_dof_pos()  # shape: (num_envs, num_steps, num_dofs)

    if flatten_steps_dim:
        return ref_dof_pos.reshape(env.num_envs, -1)
    else:
        return ref_dof_pos

def ref_joint_vel(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:

    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    ref_dof_vel = animation_term.get_dof_vel()  # shape: (num_envs, num_steps, num_dofs)

    if flatten_steps_dim:
        return ref_dof_vel.reshape(env.num_envs, -1)
    else:
        return ref_dof_vel

def ref_key_body_pos_b(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:

    animation_term: AnimationTerm = env.animation_manager.get_term(animation)

    ref_key_body_pos_b = animation_term.get_key_body_pos_b()  # shape: (num_envs, num_steps, num_key_bodies, 3)

    if flatten_steps_dim:
        return ref_key_body_pos_b.reshape(env.num_envs, -1)
    else:
        num_envs = ref_key_body_pos_b.shape[0]
        num_steps = ref_key_body_pos_b.shape[1]
        return ref_key_body_pos_b.reshape(num_envs, num_steps, -1)


def ref_root_height(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:

    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs
    ref_root_z = animation_term.get_root_pos_w()[..., 2:3]

    if flatten_steps_dim:
        return ref_root_z.reshape(num_envs, -1)
    else:
        return ref_root_z


def ref_amp_observation(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
    include_key_body_pos_b: bool = True,
    include_base_motion: bool = True,
    include_root_height: bool = True,
) -> torch.Tensor:
    """Reference Go2 AMP observation sequence from the motion data."""
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs

    ref_dof_pos = animation_term.get_dof_pos()
    num_steps = ref_dof_pos.shape[1]
    ref_dof_vel = animation_term.get_dof_vel()

    obs_terms = [ref_dof_pos]
    if include_key_body_pos_b:
        ref_key_body_pos_b = animation_term.get_key_body_pos_b()
        obs_terms.append(ref_key_body_pos_b.reshape(num_envs, num_steps, -1))
    if include_base_motion:
        ref_root_vel_b = ref_root_lin_vel_b(env, animation, flatten_steps_dim=False)
        ref_root_ang_vel_obs = ref_root_ang_vel_b(env, animation, flatten_steps_dim=False)
        obs_terms.extend([ref_root_vel_b, ref_root_ang_vel_obs])
    obs_terms.append(ref_dof_vel)
    if include_root_height:
        ref_root_z = animation_term.get_root_pos_w()[..., 2:3]
        obs_terms.append(ref_root_z)
    obs = torch.cat(obs_terms, dim=-1)

    if flatten_steps_dim:
        return obs.reshape(num_envs, -1)
    else:
        return obs

def root_local_rot_tan_norm(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    root_quat = robot.data.root_quat_w
    yaw_quat = math_utils.yaw_quat(root_quat)

    root_quat_local = math_utils.quat_mul(math_utils.quat_conjugate(yaw_quat), root_quat)

    root_rotm_local = math_utils.matrix_from_quat(root_quat_local)
    # use the first and last column of the rotation matrix as the tangent and normal vectors
    tan_vec = root_rotm_local[:, :, 0]  # (N, 3)
    norm_vec = root_rotm_local[:, :, 2]  # (N, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (N, 6)

    return obs


def ref_root_local_rot_tan_norm(
    env: ManagerBasedAnimationEnv,
    animation: str,
    flatten_steps_dim: bool = True,
) -> torch.Tensor:

    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    num_envs = env.num_envs

    ref_root_quat = animation_term.get_root_quat() # shape: (num_envs, num_steps, 4)
    ref_yaw_quat = math_utils.yaw_quat(ref_root_quat)
    ref_root_quat_local = math_utils.quat_mul(
        math_utils.quat_conjugate(ref_yaw_quat), ref_root_quat
    )  # shape: (num_envs, num_steps, 4)
    ref_root_rotm_local = math_utils.matrix_from_quat(ref_root_quat_local) # shape: (num_envs, num_steps, 3, 3)

    tan_vec = ref_root_rotm_local[:, :, :, 0]  # (num_envs, num_steps, 3)
    norm_vec = ref_root_rotm_local[:, :, :, 2]  # (num_envs, num_steps, 3)
    obs = torch.cat([tan_vec, norm_vec], dim=-1)  # (num_envs, num_steps, 6)

    if flatten_steps_dim:
        return obs.reshape(num_envs, -1)
    else:
        return obs

# tsdepth rewards
def scalar_rigid_friction_mean(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    mats = asset.root_physx_view.get_material_properties().to(env.device)
    mu_s = mats[:, :, 0].mean(dim=1)
    mu_d = mats[:, :, 1].mean(dim=1)
    return ((mu_s + mu_d) * 0.5).unsqueeze(-1)

def body_mass_scale(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    m = asset.root_physx_view.get_masses().to(env.device)
    d = asset.data.default_mass.to(env.device)
    ratio = m[:, asset_cfg.body_ids] / (d[:, asset_cfg.body_ids] + 1e-08)
    return ratio.mean(dim=-1, keepdim=True)

def body_com_pos_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # asset.data.body_com_pos_b: [num_envs, num_bodies, 3]
    com_b = asset.data.body_com_pos_b[:, asset_cfg.body_ids, :]
    return com_b.reshape(com_b.shape[0], -1)

def last_push_delta_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, '_ts_depth_push_xy'):
        return torch.zeros(env.num_envs, 2, device=env.device)
    return env._ts_depth_push_xy.to(device=env.device) # type:ignore

def joint_stiffness_scale(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot', joint_names=['.*'])) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    s = asset.data.joint_stiffness
    d = asset.data.default_joint_stiffness
    if s is None or d is None:
        return torch.ones(env.num_envs, asset.num_joints, device=env.device)
    s = s[:, asset_cfg.joint_ids]
    d = d[:, asset_cfg.joint_ids]
    return s / (d + 1e-08)

def joint_damping_scale(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot', joint_names=['.*'])) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    damp = asset.data.joint_damping
    d0 = asset.data.default_joint_damping
    if damp is None or d0 is None:
        return torch.ones(env.num_envs, asset.num_joints, device=env.device)
    damp = damp[:, asset_cfg.joint_ids]
    d0 = d0[:, asset_cfg.joint_ids]
    return damp / (d0 + 1e-08)

def height_relative_to_feet(env: ManagerBasedRLEnv, sensor_names: list[str], asset_cfg: SceneEntityCfg=SceneEntityCfg('robot'), clip: tuple[float, float] | None=(-1.0, 1.0)) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    parts = []
    for (i, name) in enumerate(sensor_names):
        sensor: RayCaster = env.scene.sensors[name] # type:ignore
        hits_z = sensor.data.ray_hits_w[..., 2]  # sensor.data.ray_hits_w 形状: [num_envs, num_rays, 3]
        rel = foot_z[:, i:i + 1] - hits_z
        if clip is not None:
            rel = rel.clamp(min=clip[0], max=clip[1])
        parts.append(rel)
    return torch.cat(parts, dim=-1)

def normal_vector_around_feet(env: ManagerBasedRLEnv, sensor_names: list[str]) -> torch.Tensor:
    parts = []
    for name in sensor_names:
        sensor: RayCaster = env.scene.sensors[name] #type:ignore
        hits = sensor.data.ray_hits_w # sensor.data.ray_hits_w 形状: [num_envs, num_rays, 3]
        # 0 1 2
        # 3 4 5
        # 6 7 8 (xy order)
        p0 = hits[:, 0]
        p1 = hits[:, 2]
        p2 = hits[:, 6]
        valid = torch.isfinite(p0).all(dim=-1) & torch.isfinite(p1).all(dim=-1) & torch.isfinite(p2).all(dim=-1)
        v1 = p1 - p0
        v2 = p2 - p0
        normal = torch.cross(v1, v2, dim=-1)
        normal = normal / torch.norm(normal, dim=-1, keepdim=True).clamp_min(1e-06)
        default = torch.zeros_like(normal)
        default[..., 2] = 1.0
        normal = torch.where(valid.unsqueeze(-1), normal, default)
        parts.append(normal)
    return torch.cat(parts, dim=-1)

def base_roll_pitch(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return base roll and pitch in radians, shape [num_envs, 2]."""

    asset: Articulation = env.scene[asset_cfg.name]

    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)

    return torch.stack((roll, pitch), dim=-1)

class ExtremeParkourProprioHistoryFrame(ManagerTermBase):
    """Build one masked 53-D proprio frame for the 10-frame history."""

    def __init__(
        self,
        cfg: ObservationTermCfg,
        env: ManagerBasedEnv,
    ) -> None:
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]

        if isinstance(asset_cfg.joint_ids, slice):
            raise ValueError(
                "ExtremeParkourProprioHistoryFrame requires 12 explicit joint IDs."
            )

        if len(asset_cfg.joint_ids) != 12:
            raise ValueError(
                "ExtremeParkourProprioHistoryFrame expected 12 joints, "
                f"but resolved {len(asset_cfg.joint_ids)}."
            )

        if isinstance(sensor_cfg.body_ids, slice):
            raise ValueError(
                "ExtremeParkourProprioHistoryFrame requires four explicit foot bodies."
            )

        if len(sensor_cfg.body_ids) != 4:
            raise ValueError(
                "ExtremeParkourProprioHistoryFrame expected four feet, "
                f"but resolved {len(sensor_cfg.body_ids)}."
            )

        self._joint_ids = list(asset_cfg.joint_ids)
        self._foot_body_ids = list(sensor_cfg.body_ids)

        self._last_contact = torch.zeros(
            env.num_envs,
            4,
            dtype=torch.bool,
            device=env.device,
        )

    def reset(self, env_ids=None) -> None:
        """Reset the one-step contact hysteresis state."""

        if env_ids is None:
            self._last_contact.zero_()
        else:
            self._last_contact[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        command_name: str,
        action_name: str,
        contact_threshold: float = 2.0,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore

        # 0:3 -- base angular velocity, scaled exactly like current proprio.
        base_ang_vel = asset.data.root_ang_vel_b * 0.25

        # 3:5 -- base roll and pitch.
        roll, pitch, _ = math_utils.euler_xyz_from_quat(
            asset.data.root_quat_w
        )
        base_roll_pitch = torch.stack((roll, pitch), dim=-1)

        # 5:13 -- eight-dimensional waypoint/navigation command.
        navigation = env.command_manager.get_command(command_name) # type:ignore

        if navigation.shape[-1] != 8:
            raise RuntimeError(
                "Expected an 8-D waypoint command, "
                f"but received shape {tuple(navigation.shape)}."
            )

        # 13:25 -- joint position relative to default pose.
        joint_pos = (
            asset.data.joint_pos[:, self._joint_ids]
            - asset.data.default_joint_pos[:, self._joint_ids]
        )

        # 25:37 -- joint velocity relative to default velocity.
        joint_vel = (
            asset.data.joint_vel[:, self._joint_ids]
            - asset.data.default_joint_vel[:, self._joint_ids]
        ) * 0.05

        # 37:49 -- previous/raw policy action.
        last_action = env.action_manager.get_term(action_name).raw_actions

        if last_action.shape[-1] != 12:
            raise RuntimeError(
                "Expected a 12-D action, "
                f"but received shape {tuple(last_action.shape)}."
            )

        # 49:53 -- one-step hysteresis-filtered foot contacts.
        current_contact = (
            torch.linalg.vector_norm(
                contact_sensor.data.net_forces_w[ # type:ignore
                    :, self._foot_body_ids, :
                ],
                dim=-1,
            )
            > contact_threshold
        )

        filtered_contact = torch.logical_or(
            current_contact,
            self._last_contact,
        )

        self._last_contact.copy_(current_contact)

        foot_contacts = filtered_contact.float() - 0.5

        history_frame = torch.cat(
            (
                base_ang_vel,
                base_roll_pitch,
                navigation,
                joint_pos,
                joint_vel,
                last_action,
                foot_contacts,
            ),
            dim=-1,
        )

        if history_frame.shape[-1] != 53:
            raise RuntimeError(
                "Extreme Parkour history frame must contain 53 values, "
                f"but received shape {tuple(history_frame.shape)}."
            )

        # Official Extreme Parkour logic masks the two target-yaw fields
        # only in the history copy. Current proprio still keeps them visible.
        history_frame[:, 6:8] = 0.0

        return history_frame

class extreme_parkour_foot_contacts(ManagerTermBase):
    """Return four hysteresis-filtered foot contacts encoded as -0.5/+0.5."""

    def __init__(
        self,
        cfg: ObservationTermCfg,
        env: ManagerBasedEnv,
    ) -> None:
        super().__init__(cfg, env)

        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        if isinstance(sensor_cfg.body_ids, slice):
            raise ValueError("extreme_parkour_foot_contacts requires four explicit foot bodies.")

        if len(sensor_cfg.body_ids) != 4:
            raise ValueError(f"Expected four foot bodies, but resolved {len(sensor_cfg.body_ids)}.")

        self._last_contact = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._last_contact.zero_()
        else:
            self._last_contact[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg,
        threshold: float = 2.0,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type:ignore

        current_contact = (
            torch.linalg.vector_norm(
                contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], # type:ignore
                dim=-1,
            )
            > threshold
        )

        filtered_contact = torch.logical_or(current_contact, self._last_contact)

        self._last_contact.copy_(current_contact)

        return filtered_contact.float() - 0.5

class ExtremeParkourMassComParams(ManagerTermBase):
    """Return added base mass and base CoM offset, shape [num_envs, 4]."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv,) -> None:
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: Articulation = env.scene[asset_cfg.name]

        if isinstance(asset_cfg.body_ids, slice):
            raise ValueError( "ExtremeParkourMassComParams requires one explicit base body.")

        if len(asset_cfg.body_ids) != 1:
            raise ValueError(
                "ExtremeParkourMassComParams expected exactly one base "
                f"body, but resolved {len(asset_cfg.body_ids)}."
            )

        self._asset = asset
        self._body_ids = list(asset_cfg.body_ids)

        self._default_mass = asset.data.default_mass[:, self._body_ids].to(env.device).clone()
        self._default_com = asset.root_physx_view.get_coms().to(env.device)[:, self._body_ids, :3].clone()

    def __call__(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg,) -> torch.Tensor:
        current_mass = self._asset.root_physx_view.get_masses().to(env.device)[:, self._body_ids]
        current_com = self._asset.root_physx_view.get_coms().to(env.device)[:, self._body_ids, :3]

        added_mass = current_mass - self._default_mass
        com_offset = current_com - self._default_com

        return torch.cat((added_mass, com_offset.reshape(env.num_envs, 3)), dim=-1)

class ExtremeParkourActuatorGainOffset(ManagerTermBase):
    """Return task actuator gain_multiplier - 1 in policy joint order."""

    _SUPPORTED_GAINS = ("stiffness", "damping")

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv,) -> None:
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        actuator_name: str = cfg.params["actuator_name"] # type:ignore
        gain_name: str = cfg.params["gain_name"]         # type:ignore

        if gain_name not in self._SUPPORTED_GAINS:
            raise ValueError(
                f"Unsupported actuator gain '{gain_name}'. "
                f"Expected one of {self._SUPPORTED_GAINS}."
            )

        asset: Articulation = env.scene[asset_cfg.name]

        if actuator_name not in asset.actuators:
            raise ValueError(
                f"Actuator '{actuator_name}' does not exist. "
                f"Available actuators: {tuple(asset.actuators.keys())}."
            )

        if isinstance(asset_cfg.joint_ids, slice):
            raise ValueError(
                "ExtremeParkourActuatorGainOffset requires explicit "
                "policy joint IDs."
            )

        self._asset = asset
        self._actuator = asset.actuators[actuator_name]
        self._gain_name = gain_name

        policy_joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.long, device=env.device,)

        # actuator.joint_indices maps actuator-local channels to global articulation joint IDs.
        """
        asset_cfg.joint_ids = [
            3, 4, 5,     # FR
            0, 1, 2,     # FL
            9, 10, 11,   # RR
            6, 7, 8,     # RL
        ]
        """
        if isinstance(self._actuator.joint_indices, slice):
            actuator_joint_ids = torch.arange(
                asset.num_joints,
                dtype=torch.long,
                device=env.device,
            )[self._actuator.joint_indices]
        else:
            actuator_joint_ids = self._actuator.joint_indices.to(device=env.device,dtype=torch.long,)

        matches = policy_joint_ids.unsqueeze(1) == actuator_joint_ids.unsqueeze(0)

        if not torch.all(matches.sum(dim=1) == 1):
            raise ValueError("Not every policy joint belongs to the selected actuator.")

        # One local actuator index for every policy joint, preserving:
        # FR, FL, RR, RL.
        # policy_joint_ids：policy 顺序下，每个关节在 articulation 中的全局 ID
        # _actuator_local_ids：policy 顺序下，每个关节在当前 actuator Tensor 中的局部列索引
        self._actuator_local_ids = matches.to(dtype=torch.long).argmax(dim=1)

        default_gain_name = f"default_joint_{gain_name}"
        default_gain = getattr(asset.data, default_gain_name,)[:, policy_joint_ids]

        if torch.any(default_gain <= 0.0):
            raise ValueError(
                f"Default joint {gain_name} must be positive to "
                "compute multiplicative offsets."
            )

        self._default_gain = default_gain.clone()

    def __call__(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, actuator_name: str, gain_name: str,) -> torch.Tensor:
        current_gain = getattr(self._actuator, self._gain_name,)[:, self._actuator_local_ids]

        gain_multiplier = current_gain / self._default_gain

        # Official Extreme Parkour passes motor_strength - 1.
        # A nominal gain therefore produces 0 rather than 1.
        return gain_multiplier - 1.0

def links_contact_binary(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float=1.0) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type:ignore
    # [num_envs, history_length, num_bodies(机器人本身的总刚体数), 3]
    net_forces = contact_sensor.data.net_forces_w_history
    # [num_envs, history_length, num_bodies, 3] -> [num_envs, history_length, num_bodies] -> [num_envs, num_bodies]
    is_contact = torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold # type:ignore
    return is_contact.float()

def depth_image_camera(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, max_depth: float=3.0, data_type: str='distance_to_image_plane') -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    depth = sensor.data.output[data_type] # [num_envs, H, W, 1]
    if depth.dim() == 4 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    # 当机器人的相机朝上看仰望天空，或者朝地平线看射向无尽的虚空时，因为射线没有击中任何物理碰撞体（Mesh），
    # 物理引擎会返回 NaN（非数）或 inf（无穷大）
    depth = torch.nan_to_num(depth, nan=max_depth, posinf=max_depth, neginf=max_depth)
    depth = depth.clamp(min=0.0, max=max_depth) / max_depth
    return depth.flatten(start_dim=1)


#=======================CTS-MOSE=====================================#
def go2_height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    offset: float = 0.5,
    clip: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name] # type:ignore

    base_z = asset.data.root_pos_w[:, 2].unsqueeze(1)
    hit_z = sensor.data.ray_hits_w[..., 2]
    safe_hit_z = torch.where(torch.isfinite(hit_z), hit_z, base_z - offset)

    heights = base_z - offset - safe_hit_z
    return torch.clamp(heights, clip[0], clip[1])

def go2_foot_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name] # type:ignore
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :] # type:ignore
    return torch.linalg.norm(contact_forces, dim=-1) * 1.0e-3

def go2_joint_torques_normalized(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    torque_limits = torch.abs(asset.data.joint_effort_limits[:, asset_cfg.joint_ids]).clamp(min=1.0e-6)
    return torques / torque_limits

def go2_joint_acc_legacy_scaled(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return -asset.data.joint_acc[:, asset_cfg.joint_ids] * 1.0e-4
