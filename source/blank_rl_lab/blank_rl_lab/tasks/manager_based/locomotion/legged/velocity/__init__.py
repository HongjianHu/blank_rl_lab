import gymnasium as gym
from . import agents

gym.register(
    id="Go2-velocity-v0",
    entry_point=f"{__name__}.aer_go2_env:AERManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_demo_velocity:GO2RobotDemoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Go2-rough-velocity-v0",
    entry_point=f"{__name__}.aer_go2_env:AERManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_rough_velocity:GO2RobotRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Go2-ts-velocity-v0",
    entry_point=f"{__name__}.aer_go2_env:AERManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_teacher_student_cfg:GO2RobotTsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.go2_ts_rsl_rl:Go2FlatDistillRunnerCfg",
    },
)

gym.register(
    id="Go2-AMP-velocity-v0",
    entry_point="blank_rl_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_go2_velocity:Go2RobotAMPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.go2_rsl_rl_ppo_amp_cfg:Go2RslRlOnPolicyRunnerAmpCfg",
    },
)

gym.register(
    id="Go2-AMP-rough-velocity-v0",
    entry_point="blank_rl_lab.envs:ManagerBasedAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_go2_rough_velocity:GO2RobotAMPRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.go2_rsl_rl_amp_rough_cfg:Go2RslRlOnPolicyRunnerAmpRoughCfg",
    },
)

gym.register(
    id="Go2-TSDepth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":f"{__name__}.tsdepth_go2_env_cfg:Go2TSDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.go2_rsl_rl_tsdepth_cfg:Go2TSDepthRunnerCfg",
    },
)

gym.register(
    id="Go2-CTS-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":f"{__name__}.go2_cts_moe_env_cfg:CTSMoeRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.go2_rsl_rl_cts_cfg:Go2CtsRunnerCfg",
    },
)

gym.register(
    id="Go2-CTSMoe-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":f"{__name__}.go2_cts_moe_env_cfg:CTSMoeRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.go2_rsl_rl_ctsmoe_cfg:Go2MoENGCtsRunnerCfg",
    },
)

gym.register(
    id="Go2-DWAQ-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":f"{__name__}.go2_dwaq_env_cfg:GO2RobotDWAQEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.go_rsl_rl_dwaq_cfg:Go2FlatDWAQRunnerCfg",
    },
)

gym.register(
    id="Go2-ExtremeParkour-Teacher-v0",
    entry_point=f"{__name__}.go2_extreme_parkour_env_cfg:ExtremeParkourManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.go2_extreme_parkour_env_cfg:Go2ExtremeParkourTeacherEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
        f"{agents.__name__}.go2_extreme_parkour_rsl_rl_cfg:Go2ExtremeParkourTeacherRunnerCfg"
        ),
    },
)
