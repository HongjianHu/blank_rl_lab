from __future__ import annotations

from dataclasses import field
from typing import ClassVar

from isaaclab.utils import configclass


@configclass
class MotionDataTermCfg:
    """Configuration for a single motion data term.

    Each term loads a set of AMP-format motion files from a directory
    and provides sampling / interpolation methods used by the discriminator
    and by AnimationTerm for reference state initialization.

    Corresponds to:
        - Go2AMPCfg.env.amp_motion_files
        - AMPLoader.__init__() 中 motion_files / time_between_frames
    """

    # 参考运动文件目录, 自动发现目录下所有 .txt 文件
    # 对应 MOTION_FILES = glob('datasets/go2_motion/*')
    motion_data_dir: str = "datasets/go2_motion"

    # 文件 → 权重映射, 空则每个文件取自身的 "MotionWeight"
    # 对应 AMPLoader.trajectory_weights
    motion_data_weights: dict[str, float] = field(default_factory=dict)

    # Expert transition spacing. If left as None, the runner should pass dt to
    # sample_amp_transitions() / feed_forward_generator().
    # 对应 AMPLoader.__init__(time_between_frames=env.dt)
    time_between_frames: float | None = None

    # 判别器预加载过渡对数量 (runner 侧使用, 这里只是透传)
    # 对应 Go2AMPCfgPPO.runner.amp_num_preload_transitions
    num_preload_transitions: int = 2_000_000

    # 判别器隐藏层维度
    # 对应 Go2AMPCfgPPO.runner.amp_discr_hidden_dims
    discr_hidden_dims: tuple[int, ...] = (1024, 512)

    # AMP style reward 系数
    # 对应 Go2AMPCfgPPO.runner.amp_reward_coef
    amp_reward_coef: float = 0.2

    # 任务 reward 与 style reward 插值系数
    # 对应 Go2AMPCfgPPO.runner.amp_task_reward_lerp
    amp_task_reward_lerp: float = 0.8

    # AMP 经验回放 buffer 大小
    # 对应 Go2AMPCfgPPO.algorithm.amp_replay_buffer_size
    amp_replay_buffer_size: int = 1_000_000
