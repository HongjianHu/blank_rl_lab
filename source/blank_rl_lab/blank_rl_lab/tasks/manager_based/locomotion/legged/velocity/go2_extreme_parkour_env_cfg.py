"""Manager-based environment configuration for the Go2 Extreme Parkour task."""

from __future__ import annotations

import torch
import isaaclab.sim as sim_utils # 地形物理材质、视觉材质、灯光等仿真配置
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg # 声明灯光一类非关节资产
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv, VecEnvStepReturn
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns # 定义射线在机器人周围的二维排列
from isaaclab.utils import configclass

from blank_rl_lab.assets.robot.unitree import UNITREE_GO2_CFG
from blank_rl_lab.terrains.parkour import (
    EXTREME_PARKOUR_TRAINING_TERRAINS_CFG,
    ExtremeParkourTerrainImporterCfg,
)

from blank_rl_lab.tasks.manager_based.locomotion.legged.velocity import mdp
# Extreme Parkour runs physics at 200 Hz and updates the policy every four
# physics steps, giving a 50 Hz policy frequency.
EXTREME_PARKOUR_SIM_DT = 0.005
EXTREME_PARKOUR_DECIMATION = 4
EXTREME_PARKOUR_POLICY_DT = EXTREME_PARKOUR_SIM_DT * EXTREME_PARKOUR_DECIMATION

# The Extreme Parkour implementation refreshes terrain heights
# every five policy steps: 5 * 0.02 s = 0.1 s, i.e. 10 Hz.
EXTREME_PARKOUR_HEIGHT_SCAN_UPDATE_PERIOD = 5 * EXTREME_PARKOUR_POLICY_DT

# The policy outputs one normalized position-offset action for each of Go2's
# twelve leg joints.  A unit action corresponds to a 0.25 rad target offset.
EXTREME_PARKOUR_ACTION_SCALE = 0.25
EXTREME_PARKOUR_NUM_ACTIONS = 12

EXTREME_PARKOUR_POLICY_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)

EXTREME_PARKOUR_POLICY_FOOT_NAMES = (
    "FR_foot",
    "FL_foot",
    "RR_foot",
    "RL_foot",
)

EXTREME_PARKOUR_HIP_JOINT_NAMES = (
    "FR_hip_joint",
    "FL_hip_joint",
    "RR_hip_joint",
    "RL_hip_joint",
)
# Official Extreme Parkour terrain scan:
# x = [-0.45, -0.30, ..., 1.20], 12 points
# y = [-0.75, -0.60, ..., 0.75], 11 points
# total = 12 * 11 = 132 rays
_EXTREME_PARKOUR_HEIGHT_SCAN_PATTERN = patterns.GridPatternCfg(
    resolution=0.15,
    size=(1.65, 1.50),
    ordering="yx",
)

_EXTREME_PARKOUR_HEIGHT_SCAN_OFFSET = RayCasterCfg.OffsetCfg(
    pos=(0.375, 0.0, 20.0),
)

# A 3 x 3 local grid covering 10 cm x 10 cm around each foot.
# This will later be used to detect whether a foot is near a terrain edge.
_EXTREME_PARKOUR_FOOT_SCAN_PATTERN = patterns.GridPatternCfg(
    resolution=0.05,
    size=(0.10, 0.10),
    ordering="yx",
)

def _make_foot_height_scanner(foot_name: str) -> RayCasterCfg:
    """Create an independent local terrain scanner for one Go2 foot."""
    return RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{foot_name}",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="world",
        pattern_cfg=_EXTREME_PARKOUR_FOOT_SCAN_PATTERN.copy(), # type:ignore
        max_distance=2.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

# ``replace`` creates a task-local articulation configuration.  The shared
# asset in assets/robot/unitree.py remains unchanged for all existing tasks.
EXTREME_PARKOUR_GO2_CFG = UNITREE_GO2_CFG.replace( # type:ignore
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.01,
        ),
    },
)

@configclass
class ExtremeParkourSceneCfg(InteractiveSceneCfg):
    """Scene shared by the teacher and future vision-student tasks."""

    terrain = ExtremeParkourTerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=EXTREME_PARKOUR_TRAINING_TERRAINS_CFG.copy(), # type:ignore
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.18, 0.20, 0.22),
            metallic=0.65,
            roughness=0.50,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = EXTREME_PARKOUR_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=_EXTREME_PARKOUR_HEIGHT_SCAN_OFFSET,
        ray_alignment="yaw",
        pattern_cfg=_EXTREME_PARKOUR_HEIGHT_SCAN_PATTERN,
        max_distance=100.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    foot_height_scanner_FL = _make_foot_height_scanner("FL_foot")
    foot_height_scanner_FR = _make_foot_height_scanner("FR_foot")
    foot_height_scanner_RL = _make_foot_height_scanner("RL_foot")
    foot_height_scanner_RR = _make_foot_height_scanner("RR_foot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
        debug_vis=False,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1800.0,
            color=(0.8, 0.8, 0.8),
        ),
    )

@configclass
class ExtremeParkourActionsCfg:
    """Twelve normalized joint-position actions for Go2."""

    joint_pos = mdp.Go2DelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(
            EXTREME_PARKOUR_POLICY_JOINT_NAMES
        ),
        scale=EXTREME_PARKOUR_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,

        # Extreme Parkour 当前不随机化电机零位。
        randomize_motor_zero_offset=False,

        # First teacher-training phase: no delay.
        randomize_action_delay=False,
        max_delay_steps=0,
    )

# env.command_manager.get_command("waypoint"), 以后通过以下方式获取公开的 8 维 Tensor
# command_term = env.command_manager.get_term("waypoint") 获取完整类对象及其额外状态：
# command_term.current_waypoint_w, command_term.next_waypoint_w 等
@configclass
class ExtremeParkourCommandsCfg:
    """Waypoint-navigation command configuration."""

    waypoint = mdp.ExtremeParkourCommandCfg(
        asset_name="robot",
        num_waypoints=8,
        reach_threshold=0.2,
        reach_hold_time=0.1,
        forward_speed_range=(0.3, 0.8),
        parkour_flat_terrain_class=2,
        debug_vis=False,
    )

@configclass
class ExtremeParkourObservationsCfg:
    """Independent observation groups for Extreme Parkour."""

    @configclass
    class ProprioCfg(ObsGroup):
        """Official current-frame 53-D proprio observation."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25,)

        base_roll_pitch = ObsTerm(func=mdp.base_roll_pitch,)

        navigation = ObsTerm(
            func=mdp.generated_commands,
            params={
                "command_name": "waypoint",
            },
        )

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(EXTREME_PARKOUR_POLICY_JOINT_NAMES),
                    preserve_order=True,
                )
            },
            scale=1.0,
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(EXTREME_PARKOUR_POLICY_JOINT_NAMES),
                    preserve_order=True,
                )
            },
            scale=0.05,
        )

        last_action = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "joint_pos"},
        )

        foot_contacts = ObsTerm(
            func=mdp.extreme_parkour_foot_contacts, # type:ignore
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=list(EXTREME_PARKOUR_POLICY_FOOT_NAMES),
                    preserve_order=True,
                ),
                "threshold": 2.0,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    proprio: ProprioCfg = ProprioCfg()

    @configclass
    class TerrainScanCfg(ObsGroup):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.3, # 地形表面恰好位于基座下方 0.3 m
            },
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    terrain_scan: TerrainScanCfg = TerrainScanCfg()

    @configclass
    class PrivExplicitCfg(ObsGroup):
        """Official 9-D explicit privileged observation."""
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
            clip=(-50.0, 50.0),
            scale=2.0,
        )

        base_lin_vel_placeholder_1 = ObsTerm(
            func=mdp.base_lin_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
            clip=(-100.0, 100.0),
            scale=0.0,
        )

        base_lin_vel_placeholder_2 = ObsTerm(
            func=mdp.base_lin_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
            clip=(-100.0, 100.0),
            scale=0.0,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    priv_explicit: PrivExplicitCfg = PrivExplicitCfg()

    @configclass
    class PrivLatentCfg(ObsGroup):
        """Official 29-D latent privileged parameters."""

        mass_com = ObsTerm(
            func=mdp.ExtremeParkourMassComParams, #type:ignore
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["base"],
                    preserve_order=True,
                ),
            },
            clip=(-100.0, 100.0),
        )

        friction = ObsTerm(
            func=mdp.scalar_rigid_friction_mean,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
            clip=(-100.0, 100.0),
        )

        stiffness_offset = ObsTerm(
            func=mdp.ExtremeParkourActuatorGainOffset, #type:ignore
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(
                        EXTREME_PARKOUR_POLICY_JOINT_NAMES
                    ),
                    preserve_order=True,
                ),
                "actuator_name": "legs",
                "gain_name": "stiffness",
            },
            clip=(-100.0, 100.0),
        )

        damping_offset = ObsTerm(
            func=mdp.ExtremeParkourActuatorGainOffset, #type:ignore
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(
                        EXTREME_PARKOUR_POLICY_JOINT_NAMES
                    ),
                    preserve_order=True,
                ),
                "actuator_name": "legs",
                "gain_name": "damping",
            },
            clip=(-100.0, 100.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    priv_latent: PrivLatentCfg = PrivLatentCfg()

    @configclass
    class ProprioHistoryCfg(ObsGroup):
        """Ten frame-major masked proprio frames: 10 x 53 = 530."""

        history = ObsTerm(
            func=mdp.ExtremeParkourProprioHistoryFrame,  # type: ignore
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(
                        EXTREME_PARKOUR_POLICY_JOINT_NAMES
                    ),
                    preserve_order=True,
                ),
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=list(
                        EXTREME_PARKOUR_POLICY_FOOT_NAMES
                    ),
                    preserve_order=True,
                ),
                "command_name": "waypoint",
                "action_name": "joint_pos",
                "contact_threshold": 2.0,
            },
            history_length=10,
            flatten_history_dim=True,
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    proprio_history: ProprioHistoryCfg = ProprioHistoryCfg()

@configclass
class ExtremeParkourRewardsCfg:
    """Official Extreme Parkour waypoint tracking rewards."""

    tracking_goal_vel = RewTerm(
        func=mdp.extreme_parkour_tracking_goal_vel,
        weight=1.5,
        params={
            "command_name": "waypoint",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    tracking_yaw = RewTerm(
        func=mdp.extreme_parkour_tracking_yaw,
        weight=0.5,
        params={
            "command_name": "waypoint",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    lin_vel_z = RewTerm(
        func=mdp.extreme_parkour_lin_vel_z,
        weight=-1.0,
        params={
            "command_name": "waypoint",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    ang_vel_xy = RewTerm(
        func=mdp.extreme_parkour_ang_vel_xy,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    orientation = RewTerm(
        func=mdp.extreme_parkour_orientation,
        weight=-1.0,
        params={
            "command_name": "waypoint",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    dof_acc = RewTerm(
        func=mdp.ExtremeParkourJointAcceleration, # type:ignore
        weight=-2.5e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(
                    EXTREME_PARKOUR_POLICY_JOINT_NAMES
                ),
                preserve_order=True,
            ),
        },
    )

    collision = RewTerm(
        func=mdp.extreme_parkour_collision,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "base",
                    ".*_thigh",
                    ".*_calf",
                ],
                preserve_order=True,
            ),
            "threshold": 0.1,
        },
    )

    action_rate = RewTerm(
        func=mdp.extreme_parkour_action_rate,
        weight=-0.1,
    )

    delta_torques = RewTerm(
        func=mdp.ExtremeParkourTorqueChange, # type:ignore
        weight=-1.0e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(
                    EXTREME_PARKOUR_POLICY_JOINT_NAMES
                ),
                preserve_order=True,
            ),
        },
    )

    torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(
                    EXTREME_PARKOUR_POLICY_JOINT_NAMES
                ),
                preserve_order=True,
            ),
        },
    )
    # hip_pos 是这个误差的髋关节子集，但两项权重不同
    # 这会更强地约束腿向身体两侧异常张开，同时仍允许 thigh/calf 完成大幅跨越动作。
    hip_pos = RewTerm(
        func=mdp.extreme_parkour_joint_position_error,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(
                    EXTREME_PARKOUR_HIP_JOINT_NAMES
                ),
                preserve_order=True,
            ),
        },
    )

    dof_error = RewTerm(
        func=mdp.extreme_parkour_joint_position_error,
        weight=-0.04,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(
                    EXTREME_PARKOUR_POLICY_JOINT_NAMES
                ),
                preserve_order=True,
            ),
        },
    )

    feet_stumble = RewTerm(
        func=mdp.extreme_parkour_feet_stumble,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=list(
                    EXTREME_PARKOUR_POLICY_FOOT_NAMES
                ),
                preserve_order=True,
            ),
            "ratio": 4.0,
        },
    )

    feet_edge = RewTerm(
        func=mdp.ExtremeParkourFeetEdge, # type:ignore
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=list(
                    EXTREME_PARKOUR_POLICY_FOOT_NAMES
                ),
                preserve_order=True,
            ),
            "scanner_names": [
                "foot_height_scanner_FR",
                "foot_height_scanner_FL",
                "foot_height_scanner_RR",
                "foot_height_scanner_RL",
            ],
            "contact_threshold": 2.0,
            "edge_height_threshold": 0.075,
            "minimum_terrain_level": 4,
        },
    )

@configclass
class ExtremeParkourEventsCfg:
    """Official startup domain randomization for Extreme Parkour."""

    robot_material = EventTerm(
        func=mdp.ExtremeParkourRandomizeRigidBodyMaterial, # type:ignore
        mode="startup",
        params={
            # No body_names means every robot rigid body.
            "asset_cfg": SceneEntityCfg("robot"),
            "friction_range": (0.6, 2.0),
            "num_buckets": 64,
            "restitution": 0.0,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, # type:ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["base"],
                preserve_order=True,
            ),
            "mass_distribution_params": (0.0, 3.0),
            "operation": "add",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["base"],
                preserve_order=True,
            ),
            "com_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
            },
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains, # type:ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(
                    EXTREME_PARKOUR_POLICY_JOINT_NAMES
                ),
                preserve_order=True,
            ),
            "stiffness_distribution_params": (
                0.8,
                1.2,
            ),
            "damping_distribution_params": (
                0.8,
                1.2,
            ),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 8.0),
        is_global_time=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
            },
        },
    )

@configclass
class ExtremeParkourCurriculumCfg:
    """Terrain difficulty curriculum for Extreme Parkour."""

    terrain_levels = CurrTerm(
        func=mdp.extreme_parkour_terrain_levels, # type:ignore
        params={
            "command_name": "waypoint",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

@configclass
class ExtremeParkourBootstrapTerminationsCfg:
    """Minimal termination contract used before parkour failures are added."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True,)
    route_complete = DoneTerm(
        func=mdp.extreme_parkour_route_complete,
        params={
            "command_name": "waypoint",
        },
        time_out=True,
    )

@configclass
class Go2ExtremeParkourTeacherEnvCfg(ManagerBasedRLEnvCfg):
    """Bootstrap environment for the Go2 Extreme Parkour teacher task."""

    seed: int = 0
    scene: ExtremeParkourSceneCfg = ExtremeParkourSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=True,)
    observations: ExtremeParkourObservationsCfg = ExtremeParkourObservationsCfg()
    actions: ExtremeParkourActionsCfg = ExtremeParkourActionsCfg()
    rewards: ExtremeParkourRewardsCfg = ExtremeParkourRewardsCfg()
    events: ExtremeParkourEventsCfg = ExtremeParkourEventsCfg()
    terminations: ExtremeParkourBootstrapTerminationsCfg = ExtremeParkourBootstrapTerminationsCfg()
    commands: ExtremeParkourCommandsCfg = ExtremeParkourCommandsCfg()
    curriculum: ExtremeParkourCurriculumCfg = ExtremeParkourCurriculumCfg()

    only_positive_rewards: bool = True

    def __post_init__(self) -> None:
        """Configure control frequency, physics, rendering and sensors."""

        # One policy action is held for four physics steps.
        self.decimation = EXTREME_PARKOUR_DECIMATION
        self.episode_length_s = 20.0 # seconds/s
        # Physics runs at 200 Hz.
        self.sim.dt = EXTREME_PARKOUR_SIM_DT
        # GUI rendering runs once per policy step, i.e. 50 Hz.
        self.sim.render_interval = self.decimation
        # Use the terrain material as the simulation-wide default material.
        self.sim.physics_material = self.scene.terrain.physics_material
        # Increase the PhysX contact-patch buffer for thousands of robots.
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # Refresh the main terrain scan every five policy steps, i.e. at 10 Hz.
        self.scene.height_scanner.update_period = EXTREME_PARKOUR_HEIGHT_SCAN_UPDATE_PERIOD
        # Foot-local terrain scans are also consumed once per policy step.
        for scanner_name in (
            "foot_height_scanner_FL",
            "foot_height_scanner_FR",
            "foot_height_scanner_RL",
            "foot_height_scanner_RR",
        ):
            scanner_cfg = getattr(self.scene, scanner_name)
            scanner_cfg.update_period = EXTREME_PARKOUR_POLICY_DT

        # Contact forces must be updated at every physics step.
        self.scene.contact_forces.update_period = EXTREME_PARKOUR_SIM_DT

class ExtremeParkourManagerBasedRLEnv(ManagerBasedRLEnv):
    """Manager-Based RL environment with official reward-total clipping."""
    cfg: Go2ExtremeParkourTeacherEnvCfg

    def __init__(self, cfg: Go2ExtremeParkourTeacherEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self.raw_reward_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        (
            observations,
            raw_reward,
            terminated,
            time_outs,
            extras,
        ) = super().step(action)
        # 是 PyTorch 的原地数值复制,把raw_reward的数值复制到self.raw_reward_buf
        # 两个 Tensor 不会共享存储
        self.raw_reward_buf.copy_(raw_reward)

        if self.cfg.only_positive_rewards:
            self.reward_buf = torch.clamp_min(raw_reward, 0.0)
        else:
            self.reward_buf = raw_reward

        return observations, self.reward_buf, terminated, time_outs, extras