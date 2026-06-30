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
    id="Go2-AMP-velocity-v0",
    entry_point=f"{__name__}.amp_go2_velocity:AMPManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amp_go2_velocity:Go2RobotAMPEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
