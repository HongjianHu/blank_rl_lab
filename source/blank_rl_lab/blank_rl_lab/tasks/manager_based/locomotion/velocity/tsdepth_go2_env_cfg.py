# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sensors.ray_caster import RayCasterCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen

from blank_rl_lab.tasks.manager_based.locomotion.velocity import mdp
from blank_rl_lab.assets.robot.unitree import UNITREE_GO2_CFG as RobotCFG
##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

_DEPTH_PITCH_DEG = 45.0
_DEPTH_ROT_Y = math.radians(_DEPTH_PITCH_DEG) - math.pi / 2.0
_DEPTH_IMAGE_HEIGHT = 30
_DEPTH_IMAGE_WIDTH = 40
_DEPTH_FOCAL_LENGTH = 24.0 # 决定焦距
_DEPTH_HFOV_DEG = 75.0     # 决定视场范围
# 这是标准针孔相机成像公式（感光宽度 = 2×焦距×tan(半视场角)），用于让仿真相机的 FOV 与真实深度相机（比如 Intel RealSense 常见 ~70-90°水平视场）相近
_DEPTH_HORIZONTAL_APERTURE = 2.0 * _DEPTH_FOCAL_LENGTH * math.tan(math.radians(_DEPTH_HFOV_DEG / 2.0)) 
_DEPTH_MAX_DEPTH = 2.0
_GO2_CONTACT_LINKS: tuple[str, ...] = ('.*_thigh', '.*_calf', '.*_foot')
_GO2_FOOT_SENSORS: tuple[str, ...] = ('foot_height_scanner_FL', 'foot_height_scanner_FR', 'foot_height_scanner_RL', 'foot_height_scanner_RR')
_GO2_FOOT_BODIES: tuple[str, ...] = ('FL_foot', 'FR_foot', 'RL_foot', 'RR_foot')
# 指定网格点的展开/遍历顺序（先沿 x 再沿 y，还是反过来）。这个参数看起来琐碎，但很重要——它决定了光线返回的高度值被拉平成一维观测向量时的排列顺序，必须和训练策略时用的观测定义保持严格一致
_FOOT_GRID_PATTERN = patterns.GridPatternCfg(resolution=0.1, size=(0.2, 0.2), ordering='xy')
_FOOT_RAY_OFFSET = RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0))

DWAQ_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # stairs up — 20 %
        "stairs_up_26": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.07,
            step_height_range=(0.0, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_30": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.07,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_34": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.06,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # stairs down — 20 %
        "stairs_down_26": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.07,
            step_height_range=(0.0, 0.23),
            step_width=0.26,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.07,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_34": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.06,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # easy terrain — 60 %
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25, noise_range=(0.0, 0.02), noise_step=0.01, border_width=0.25
        ),
        "smooth_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.2), platform_width=2.0, inverted=False
        ),
        "rough_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.2), platform_width=2.0, inverted=True
        ),
        "discrete": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.0, 0.1), platform_width=2.0
        ),
    },
)

@configclass
class TSDepthSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg =  MISSING # type: ignore
    # sensors
    depth_scanner = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.45), rot=(math.cos(_DEPTH_ROT_Y / 2.0), 0.0, math.sin(_DEPTH_ROT_Y / 2.0), 0.0), convention='opengl'),
        # 用光线投射（ray casting）模拟针孔相机成像
        pattern_cfg=patterns.PinholeCameraPatternCfg(focal_length=_DEPTH_FOCAL_LENGTH, horizontal_aperture=_DEPTH_HORIZONTAL_APERTURE, vertical_aperture=None, height=_DEPTH_IMAGE_HEIGHT, width=_DEPTH_IMAGE_WIDTH),
        data_types=['distance_to_image_plane'],  # 沿相机光轴方向的平面深度"（z-depth，类似真实深度相机 RealSense 输出的格式）
        depth_clipping_behavior='max', 
        max_distance=_DEPTH_MAX_DEPTH, 
        debug_vis=True, 
        mesh_prim_paths=['/World/ground']
        )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)), # 把光线发射起点设置在机身正上方 20 米高处,由于机身在不同地形（上坡、下台阶）时实际高度会变化，把发射点放得足够高，可以保证无论机身当前姿态如何，光线永远能从地形上方开始、往下打中地形
        ray_alignment="yaw", # 光线网格（pattern）只跟随机身的偏航角（yaw）旋转
        # GridPatternCfg 在机身周围铺一个 1.8m（前后）× 0.8m（左右）、间隔 0.1m 的网格化光线阵列（约 19×9 ≈ 171 条射线）
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.8, 0.8]),# type: ignore
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    foot_height_scanner_FL = RayCasterCfg(prim_path='{ENV_REGEX_NS}/Robot/FL_foot', offset=_FOOT_RAY_OFFSET, ray_alignment='yaw', pattern_cfg=_FOOT_GRID_PATTERN, debug_vis=False, mesh_prim_paths=['/World/ground'])
    foot_height_scanner_FR = RayCasterCfg(prim_path='{ENV_REGEX_NS}/Robot/FR_foot', offset=_FOOT_RAY_OFFSET, ray_alignment='yaw', pattern_cfg=_FOOT_GRID_PATTERN, debug_vis=False, mesh_prim_paths=['/World/ground'])
    foot_height_scanner_RL = RayCasterCfg(prim_path='{ENV_REGEX_NS}/Robot/RL_foot', offset=_FOOT_RAY_OFFSET, ray_alignment='yaw', pattern_cfg=_FOOT_GRID_PATTERN, debug_vis=False, mesh_prim_paths=['/World/ground'])
    foot_height_scanner_RR = RayCasterCfg(prim_path='{ENV_REGEX_NS}/Robot/RR_foot', offset=_FOOT_RAY_OFFSET, ray_alignment='yaw', pattern_cfg=_FOOT_GRID_PATTERN, debug_vis=False, mesh_prim_paths=['/World/ground'])

    # contact sensor
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        update_period=0.0,
        track_air_time=True,
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings 
# Isaac Lab的观测管理器要求所有MDP项的输出必须是二维的[num_envs, feature_dim]
##
@configclass
class Go2TSDepthObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={'command_name': 'base_velocity'}, scale=(1.0, 1.0, 0.25))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), scale=0.05)
        actions = ObsTerm(func=mdp.last_action, scale=0.1)

        def __post_init__(self):
            self.history_length = 1
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class PrivilegedCfg(ObsGroup):
        dr_friction = ObsTerm(func=mdp.scalar_rigid_friction_mean, params={'asset_cfg': SceneEntityCfg('robot')})
        # 让策略（Policy）显式地知道当前环境里机器人被“加重”或“减轻”了多少
        dr_mass_scale = ObsTerm(func=mdp.body_mass_scale, params={'asset_cfg': SceneEntityCfg('robot', body_names='base')})
        dr_com_b = ObsTerm(func=mdp.body_com_pos_b, params={'asset_cfg': SceneEntityCfg('robot', body_names='base')})
        dr_push_xy = ObsTerm(func=mdp.last_push_delta_xy)
        dr_kp_scale = ObsTerm(func=mdp.joint_stiffness_scale, params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'])})
        dr_kd_scale = ObsTerm(func=mdp.joint_damping_scale, params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'])})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_relative_to_feet = ObsTerm(
            func=mdp.height_relative_to_feet, 
            params={'sensor_names': list(_GO2_FOOT_SENSORS), 
                    'asset_cfg': SceneEntityCfg('robot', body_names=list(_GO2_FOOT_BODIES)), 
                    'clip': (-1.0, 1.0)}
                    )
        normal_vector_around_feet = ObsTerm(func=mdp.normal_vector_around_feet, params={'sensor_names': list(_GO2_FOOT_SENSORS)})
        link_contact_states = ObsTerm(func=mdp.links_contact_binary, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names=list(_GO2_CONTACT_LINKS)), 'threshold': 1.0})
        # 通过引入 offset，函数把平地的输出信号归一化为了0.0。这就给神经网络提供了一个极佳的基准线
        height_scan = ObsTerm(func=mdp.height_scan, params={'sensor_cfg': SceneEntityCfg('height_scanner'), 'offset': 0.4}, clip=(-1.0, 1.0), scale=2.0)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={'command_name': 'base_velocity'}, scale=(1.0, 1.0, 0.25))
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action, scale=0.1)
        dr_friction = ObsTerm(func=mdp.scalar_rigid_friction_mean, params={'asset_cfg': SceneEntityCfg('robot')})
        dr_mass_scale = ObsTerm(func=mdp.body_mass_scale, params={'asset_cfg': SceneEntityCfg('robot', body_names='base')})
        dr_com_b = ObsTerm(func=mdp.body_com_pos_b, params={'asset_cfg': SceneEntityCfg('robot', body_names='base')})
        dr_push_xy = ObsTerm(func=mdp.last_push_delta_xy)
        dr_kp_scale = ObsTerm(func=mdp.joint_stiffness_scale, params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'])})
        dr_kd_scale = ObsTerm(func=mdp.joint_damping_scale, params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'])})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        link_contact_states = ObsTerm(func=mdp.links_contact_binary, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names=list(_GO2_CONTACT_LINKS)), 'threshold': 1.0})
        height_relative_to_feet = ObsTerm(
            func=mdp.height_relative_to_feet, 
            params={'sensor_names': list(_GO2_FOOT_SENSORS), 
                    'asset_cfg': SceneEntityCfg('robot', body_names=list(_GO2_FOOT_BODIES)), 
                    'clip': (-1.0, 1.0)}
                    )
        normal_vector_around_feet = ObsTerm(func=mdp.normal_vector_around_feet, params={'sensor_names': list(_GO2_FOOT_SENSORS)})
        height_scan = ObsTerm(func=mdp.height_scan, params={'sensor_cfg': SceneEntityCfg('height_scanner'), 'offset': 0.4}, clip=(-1.0, 1.0), scale=2.0)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DepthCfg(ObsGroup):
        depth_image = ObsTerm(func=mdp.depth_image_camera, params={'sensor_cfg': SceneEntityCfg('depth_scanner'), 'max_depth': _DEPTH_MAX_DEPTH}, noise=Unoise(n_min=-0.02, n_max=0.02))

        def __post_init__(self):
            self.history_length = 1
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()
    critic: CriticCfg = CriticCfg()
    depth: DepthCfg = DepthCfg()

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", 
                                           joint_names=[".*"], 
                                           scale=0.25, 
                                           use_default_offset=True)

@configclass
class CommandsCfg:
    base_velocity = mdp.UniformLevelVelocityCommandCfg(asset_name='robot',
                                                  resampling_time_range=(10.0, 10.0),
                                                  rel_standing_envs=0.2,  # 20% 的环境被要求原地站立不动
                                                  rel_heading_envs=1.0,   # 100% 的环境使用 heading 模式
                                                  heading_command=True,
                                                  heading_control_stiffness=0.5,
                                                  debug_vis=True,
                                                  ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
                                                      lin_vel_x=(0.0, 0.5),
                                                      lin_vel_y=(0.0, 0.0),
                                                      ang_vel_z=(-0.5, 0.5),
                                                      heading=(-math.pi, math.pi)),
                                                  limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
                                                      lin_vel_x=(0.0, 1.5),
                                                      lin_vel_y=(0.0, 0.0),
                                                      ang_vel_z=(-1.2, 1.2),
                                                      heading=(-math.pi, math.pi))
                                                      )

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material, # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),    # 静摩擦系数的采样范围
            "dynamic_friction_range": (0.6, 0.6),   # 动摩擦系数
            "restitution_range": (0.0, 0.0),        # 弹性恢复系数
            "num_buckets": 64,                      # 预采样64种材质组合，然后随机分配给每个刚体的每个碰撞形状。因为 PhysX 最多支持 64000 种材质，用桶机制避免超出限制
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add", # 在默认质量基础上加上采样值(kg)
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            # 在默认重心位置上叠加均匀采样的偏移(cm)
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),      # 施加到 base 的力范围(N)
            "torque_range": (-0.0, 0.0),    # 施加到 base 的力矩范围 (N·m)
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5), # 初始 x 位置偏移 ±0.5m
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (0, 0),
            },
        },
    )

    reset_hip_joints = EventTerm(
        func=mdp.reset_joints_by_offset, 
        mode='reset', 
        params={'asset_cfg': SceneEntityCfg('robot', joint_names='.*_hip_joint'), 'position_range': (-0.2, 0.2), 'velocity_range': (0.0, 0.0)}
        )
    
    reset_thigh_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode='reset', 
        params={'asset_cfg': SceneEntityCfg('robot', joint_names='.*_thigh_joint'), 'position_range': (-0.4, 0.4), 'velocity_range': (0.0, 0.0)}
        )
    
    reset_calf_joints = EventTerm(
        func=mdp.reset_joints_by_offset, 
        mode='reset', 
        params={'asset_cfg': SceneEntityCfg('robot', joint_names='.*_calf_joint'), 'position_range': (-0.4, 0.4), 'velocity_range': (0.0, 0.0)}
        )
    
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity_record_xy, 
        mode='interval', 
        interval_range_s=(3.0, 3.0), 
        params={'velocity_range': {'x': (-0.5, 0.5), 'y': (-0.5, 0.5)}, 'asset_cfg': SceneEntityCfg('robot')})
    
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains, # type:ignore
        mode='startup', 
        params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*']), 'stiffness_distribution_params': (0.8, 1.2), 'damping_distribution_params': (0.8, 1.2), 'operation': 'scale', 'distribution': 'uniform'}
        )

@configclass
class Go2TSDepthRewardsCfg:
    tracking_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_heading_exp, 
        weight=1.5, 
        params={'command_name': 'base_velocity', 'std': math.sqrt(0.2), 'y_error_weight': 2.0}
        )
    tracking_ang_vel = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={'command_name': 'base_velocity', 'std': math.sqrt(0.2)})
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    dof_power = RewTerm(
        func=mdp.dof_power_l1, 
        weight=-2e-05, 
        params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'])}
        )
    dof_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2e-07, params={'asset_cfg': SceneEntityCfg('robot', joint_names=['.*'])})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    action_smoothness = RewTerm(func=mdp.action_smoothness_second_order, weight=-0.01)
    
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_target, 
        weight=0.2, 
        params={'sensor_cfg': SceneEntityCfg('height_scanner'), 'asset_cfg': SceneEntityCfg('robot', body_names='.*_foot'), 'target_height': 0.08, 'foot_offset': 0.022, 'sigma': 0.01})
    feet_contact_stand_still = RewTerm(
        func=mdp.feet_contact_stand_still, 
        weight=0.1, 
        params={'command_name': 'base_velocity', 'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_foot'), 'cmd_threshold': 0.2, 'force_threshold': 10.0})
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.0, 
        params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names='.*_foot')})
    hip_pos = RewTerm(
        func=mdp.hip_pos_deviation, 
        weight=-0.15, 
        params={'asset_cfg': SceneEntityCfg('robot', joint_names='.*_hip_joint')})
    
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    collision = RewTerm(func=mdp.undesired_contacts, weight=-5.0, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names=['base', '.*_thigh', '.*_calf']), 'threshold': 1.0})

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    gravity_tilt = DoneTerm(func=mdp.gravity_too_horizontal, params={'threshold': -0.1})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel) # type: ignore
    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels, # type:ignore
        params={'reward_term_name': 'tracking_lin_vel'}
        )

##
# Environment configuration
##
@configclass
class Go2TSDepthEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    scene: TSDepthSceneCfg = TSDepthSceneCfg(num_envs=1024, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    observations: Go2TSDepthObservationsCfg = Go2TSDepthObservationsCfg()
    rewards:Go2TSDepthRewardsCfg = Go2TSDepthRewardsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        for sensor_name in ('foot_height_scanner_FL', 'foot_height_scanner_FR', 'foot_height_scanner_RL', 'foot_height_scanner_RR'):
            sensor = getattr(self.scene, sensor_name, None)
            if sensor is not None:
                sensor.update_period = self.decimation * self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None: # type:ignore
            self.scene.contact_forces.update_period = self.sim.dt # type:ignore
        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        robot = RobotCFG.replace(prim_path='{ENV_REGEX_NS}/Robot') # type:ignore
        robot.init_state.pos = (0.0, 0.0, 0.4)
        robot.init_state.joint_pos = {'.*_hip_joint': 0.0, '.*_thigh_joint': 0.8, '.*_calf_joint': -1.5}
        if 'GO2HV' in robot.actuators:
            robot.actuators['GO2HV'].stiffness = 30.0
            robot.actuators['GO2HV'].damping = 0.75
        self.scene.robot = robot
        self.events.physics_material.params['static_friction_range'] = (0.2, 1.7)
        self.events.physics_material.params['dynamic_friction_range'] = (0.2, 1.7)
        self.events.add_base_mass.params['mass_distribution_params'] = (-1.0, 2.0)
        self.events.base_com.params['com_range'] = {'x': (-0.03, 0.03), 'y': (-0.03, 0.03), 'z': (-0.03, 0.03)}
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)


