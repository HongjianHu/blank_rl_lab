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
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen

from blank_rl_lab.assets.robot.unitree import UNITREE_GO2_CFG as RobotCFG
from blank_rl_lab.tasks.manager_based.locomotion.velocity import mdp
from blank_rl_lab.tasks.manager_based.locomotion.velocity.amp_go2_velocity import Go2RobotAMPEnvCfg
from blank_rl_lab.tasks.manager_based.locomotion.velocity.amp_go2_velocity import RewardsCfg as AmpRewardsCfg

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=None,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
        "smooth_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.04),
            platform_width=3.0,
            border_width=0.25,
        ),
        "smooth_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.04),
            platform_width=3.0,
            border_width=0.25,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.0, 0.05),
            noise_step=0.005,
            downsampled_scale=0.2,
            border_width=0.25,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.05, 0.07),
            step_width=0.31,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.07),
            step_width=0.31,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.20,
            obstacle_height_mode="choice",
            obstacle_width_range=(1.0, 2.0),
            obstacle_height_range=(0.05, 0.07),
            num_obstacles=20,
            platform_width=3.0,
            border_width=0.25,
        ),
    },
)

@configclass
class RoughSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
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
    robot: ArticulationCfg = RobotCFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]), # type: ignore
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.5, 0.5)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 2.0), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.8, 0.8)
        ),
    )

@configclass
class RoughRewardsCfg(AmpRewardsCfg):
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)

    feet_slip = RewTerm(
        func=mdp.feet_slip,
        weight=-0.04,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    base_height = RewTerm(
        func=mdp.safe_base_height_l2,
        weight=-2.0,
        params={
            "target_height": 0.3,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel) # type: ignore

@configclass
class GO2RobotAMPRoughEnvCfg(Go2RobotAMPEnvCfg):
    """Configuration for the GO2 robot environment with rough terrain."""
    scene: RoughSceneCfg = RoughSceneCfg(num_envs=4096, env_spacing=2.5)

    commands: CommandsCfg = CommandsCfg()
    rewards: RoughRewardsCfg = RoughRewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
