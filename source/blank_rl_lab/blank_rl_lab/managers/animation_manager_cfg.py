# ============ animation_manager_cfg.py ============

from __future__ import annotations

from dataclasses import MISSING
from typing import ClassVar, Sequence

from isaaclab.utils import configclass


@configclass
class AnimationTermCfg:
    """Configuration for a single animation term.

    Controls how robot states are initialized from AMP reference
    motion data at episode reset.
    """

    # -- 功能开关 --
    # 是否启用参考状态初始化
    # 对应 cfg.env.reference_state_initialization = True
    enable: bool = True

    # 使用参考运动初始化的概率 [0.0, 1.0]
    #   1.0 = 全部从参考运动初始化 (等价于 prob = 999)
    #   0.0 = 全部用默认随机初始化
    # 对应 cfg.env.reference_state_initialization_prob
    probability: float = 1.0

    # -- 数据绑定 --
    # 引用 MotionDataManager 中的哪个 motion data term
    # 对应: env.amp_loader 的数据来源
    motion_data_term: str = "motion_dataset"

    # -- 状态分量 --
    # 重置时从参考帧中取哪些分量写入 sim
    #
    #   "root_pos_w"       → reset_root_states_amp: 基座世界位置    [3]
    #   "root_quat"        → reset_root_states_amp: 基座世界旋转    [4]
    #   "root_vel_w"       → reset_root_states_amp: 基座线速度      [3]
    #   "root_ang_vel_w"   → reset_root_states_amp: 基座角速度      [3]
    #   "dof_pos"          → reset_dofs_amp:      关节角度         [12]
    #   "dof_vel"          → reset_dofs_amp:      关节角速度       [12]
    #   "key_body_pos_b"   → 足端在基座系下位置 (校验用, 不写 sim)   [12]
    motion_data_components: ClassVar[list[str]] = [
        "root_pos_w",
        "root_quat",
        "root_vel_w",
        "root_ang_vel_w",
        "dof_pos",
        "dof_vel",
        "key_body_pos_b",
    ]

    # -- 采样策略 --
    # 重置时从轨迹中采多少帧 (当前项目只设初始姿态, 固定为 1)
    #   >0: 向前播放 N 步
    #   <0: 向后追溯 |N| 步
    #   1:  只取 1 帧设置初始姿态 (go2_amp 默认行为)
    num_steps_to_use: int = 1

    # 起始时间是否随机
    # True: 从轨迹中随机时间点采样 (go2_amp: get_full_frame_batch → 随机)
    # False: 从轨迹开头开始
    random_initialize: bool = True

    # 是否在每个 update() 重新随机拉取参考帧。
    # go2_amp 的 reference_state_initialization 只在 reset 时采样，所以默认 False。
    # True 主要用于可视化/调试，不建议用于训练初始化逻辑。
    random_fetch: bool = False

    # -- 可视化 --
    # 是否显示参考姿态 ghost robot
    enable_visualization: bool = False

    # ghost robot 相对于基座的高度偏移 [m]
    vis_root_offset: tuple[float, float, float] = (0.0, 0.0, 0.5)
