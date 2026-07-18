import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

UNITREE_MODEL_DIR = "/home/robot/workshop/Robot/IsaacLab/blank_rl_lab/blank_rl_lab/source/blank_rl_lab/blank_rl_lab/assets/robot/unitree_model"

UNITREE_GO2_CFG = ArticulationCfg(
    # spawn=UnitreeUrdfFileCfg(
    #     asset_path=f"{UNITREE_ROS_DIR}/robots/go2_description/urdf/go2_description.urdf",
    # ),
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/Go2/usd/go2.usd",
        activate_contact_sensors = True,
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*R_hip_joint": -0.0,
            ".*L_hip_joint": 0.0,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor = 0.9,
    actuators={
        # 纯仿真阶段使用 IdealPDActuatorCfg：
        #   - PD 控制 + 一刀切力矩限幅，简单、收敛快
        #   - 不引入 DCMotor 的转速-力矩衰减，避免仿真中出现不存在的"高速无力"
        # sim2real 阶段可升级为 UnitreeActuatorCfg_Go2HV (DelayedPDActuator 派生)：
        #   - 真实电机扭矩-转速曲线 (X1=13.5, X2=30, Y1=20.2, Y2=23.4)
        #   - 静态/动态摩擦力模型
        #   - 指令延迟 (模拟 EtherCAT 通信延迟)
        #   参考: unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree_actuators.py
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
        ),
        # 旧版 DCMotor 执行器配置（保留备查，切回来取消下面注释即可）:
        # "legs": DCMotorCfg(
        #     joint_names_expr=[".*"],
        #     effort_limit=23.5,
        #     saturation_effort=23.5,
        #     velocity_limit=30.0,
        #     stiffness=25.0,
        #     damping=0.5,
        #     friction=0.01,
        # ),
    },
)

"""Configuration for Unitree Go2W wheeled robot.

The Go2W is a wheeled quadruped with:
- 4 legs, each with hip + thigh + calf joints (3 DOF × 4 = 12 DOF)
- 4 wheel joints at the feet (4 DOF)
- Total: 16 DOF

Reference: https://github.com/unitreerobotics/unitree_ros
"""
UNITREE_GO2W_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UNITREE_MODEL_DIR}/Go2W/usd/go2w.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs_and_wheels": IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=23.5,
            velocity_limit=30.0,
            stiffness={
                ".*_hip_.*": 25.0,
                ".*_thigh_.*": 25.0,
                ".*_calf_.*": 25.0,
                ".*_foot_.*": 0.0,  # wheels: zero stiffness for velocity-controlled spinning
            },
            damping={
                ".*_hip_.*": 0.5,
                ".*_thigh_.*": 0.5,
                ".*_calf_.*": 0.5,
                ".*_foot_.*": 0.5,
            },
            friction=0.01,
        ),
    },
)
