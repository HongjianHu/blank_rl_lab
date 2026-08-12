"""Configurations for continuous-demo and five-family parkour terrains."""

from __future__ import annotations

from typing import Literal

from isaaclab.terrains import (
    SubTerrainBaseCfg,
    TerrainGeneratorCfg,
    TerrainImporterCfg,
)
from isaaclab.utils import configclass

from .parkour_terrain import (
    ExtremeParkourTerrainGenerator,
    ExtremeParkourTerrainImporter,
    build_parkour,
    build_parkour_flat,
    build_parkour_gap,
    build_parkour_hurdle,
    build_parkour_step,
    continuous_parkour_terrain,
    parkour_flat_terrain,
    parkour_gap_terrain,
    parkour_hurdle_terrain,
    parkour_step_terrain,
    parkour_terrain,
)


# -----------------------------------------------------------------------------
# Existing continuous mixed demonstration terrain
# -----------------------------------------------------------------------------

@configclass
class ExtremeParkourTerrainCfg(SubTerrainBaseCfg):
    """Parameters for the existing continuous mixed parkour course."""

    function = continuous_parkour_terrain  # type: ignore

    track_width: float = 1.6  # 连续赛道可行走区域宽度。
    ground_height: float = 0.0  # 普通地面的顶面高度。
    foundation_depth: float = 1.0  # 可站立实体向下延伸的深度。
    pit_floor_thickness: float = 0.10  # 坑底实体自身厚度。
    spawn_x: float = 1.0  # 机器人出生点的局部 x 坐标。

    start_length: float = 2.0  # 起点平台长度。
    hurdle_section_length: float = 3.0  # 栏障路段总长度。
    step_section_length: float = 4.0  # 台阶路段总长度。
    gap_section_length: float = 3.0  # 沟壑路段总长度。
    stone_section_length: float = 4.0  # 离散踏石路段总长度。
    ramp_section_length: float = 2.0  # 上下坡路段总长度。
    finish_length: float = 2.0  # 终点平台长度。

    hurdle_runup: float = 1.25  # 栏障前的助跑距离。
    hurdle_depth: float = 0.25  # 栏障沿 x 方向的厚度。
    hurdle_height_range: tuple[float, float] = (0.12, 0.32)  # 简单到困难的栏障高度。

    step_tread_length: float = 0.8  # 单级台阶踏面长度。
    step_height_range: tuple[float, float] = (0.08, 0.18)  # 简单到困难的单级高度。

    gap_length_range: tuple[float, float] = (0.25, 0.70)  # 简单到困难的沟壑宽度。

    num_stones: int = 4  # 踏石数量。
    stone_length_range: tuple[float, float] = (0.72, 0.45)  # 简单到困难的踏石长度。
    stone_width_range: tuple[float, float] = (1.25, 0.75)  # 简单到困难的踏石宽度。
    stone_height_range: tuple[float, float] = (0.02, 0.12)  # 踏石随机高度上限。
    stone_lateral_offset_range: tuple[float, float] = (0.0, 0.28)  # 踏石左右错位幅度。

    ramp_height_range: tuple[float, float] = (0.12, 0.35)  # 简单到困难的坡顶高度。


def make_parkour_terrain_cfg(
    *,
    num_rows: int = 10,
    num_cols: int = 20,
    curriculum: bool = True,
    difficulty_range: tuple[float, float] = (0.0, 1.0),
    seed: int = 0,
    color_scheme: Literal["height", "random", "none"] = "height",
) -> TerrainGeneratorCfg:
    """Create the existing continuous mixed-course generator config."""
    return TerrainGeneratorCfg(
        seed=seed,
        curriculum=curriculum,
        size=(20.0, 4.0),
        border_width=5.0,
        border_height=1.0,
        num_rows=num_rows,
        num_cols=num_cols,
        color_scheme=color_scheme,
        difficulty_range=difficulty_range,
        use_cache=False,
        sub_terrains={
            "continuous_parkour": ExtremeParkourTerrainCfg(
                proportion=1.0,
            ),
        },
    )


PARKOUR_TERRAINS_CFG = make_parkour_terrain_cfg()

# -----------------------------------------------------------------------------
# Five-family curriculum training terrains
# -----------------------------------------------------------------------------

@configclass
class ParkourBaseTerrainCfg(SubTerrainBaseCfg):
    """Parameters shared by all five terrain families."""

    ground_height: float = 0.0  # 普通可行走表面的顶面高度。
    foundation_depth: float = 1.0  # 地形主体从地面向下延伸的深度。
    pit_floor_thickness: float = 0.10  # 有限坑底的实体厚度。

    horizontal_scale: float = 0.05  # x/y 尺寸的离散分辨率。
    vertical_scale: float = 0.005  # z 高度的离散分辨率。

    platform_length: float = 2.5  # 每条路线的起点平台长度。
    spawn_x: float = 1.0  # 机器人出生点的局部 x 坐标。
    finish_goal_margin: float = 0.5  # 最终目标距 tile 末端的安全距离。

    num_waypoints: int = 8  # 每条路线的目标点数量。
    num_obstacles: int = 6  # 中间障碍数量，必须等于 waypoint 数量减 2。


@configclass
class ParkourTerrainCfg(ParkourBaseTerrainCfg):
    """Laterally alternating inclined stepping platforms."""

    function = parkour_terrain  # IsaacLab 标准地形函数，只返回 mesh 和 origin。
    metadata_function = build_parkour  # 自定义生成函数，额外返回 waypoint。
    terrain_class = 0  # 地形类别编号：倾斜踏石。

    stone_length_lower_range: tuple[float, float] = (0.9, 0.6)  # 踏石长度采样下界随难度变化。
    stone_length_upper_range: tuple[float, float] = (1.0, 0.8)  # 踏石长度采样上界随难度变化。

    stone_gap_min: float = -0.1  # 相邻踏石投影允许的最小边缘间隔。
    stone_gap_max_range: tuple[float, float] = (0.1, 0.4)  # 最大边缘间隔随难度变化。

    lateral_offset_min: float = 0.2  # 踏石中心的最小横向偏移。
    lateral_offset_max_range: tuple[float, float] = (0.3, 0.4)  # 最大横向偏移随难度变化。

    stone_width: float = 1.0  # 普通及最后踏石沿 y 方向的宽度。
    last_stone_length: float = 1.6  # 最后一块加长踏石的 x 长度。

    incline_height_range: tuple[float, float] = (0.0, 0.25)  # 普通踏石横向倾斜高差。
    last_incline_height_range: tuple[float, float] = (0.1, 0.25)  # 最后踏石横向倾斜高差。

    final_platform_gap: float = 0.05  # 最后踏石边缘到终点平台的缝隙。
    pit_depth_range: tuple[float, float] = (0.2, 1.0)  # 坑底深度采样范围。


@configclass
class ParkourHurdleTerrainCfg(ParkourBaseTerrainCfg):
    """Repeated hurdle course."""

    function = parkour_hurdle_terrain  # IsaacLab 标准栏障地形入口。
    metadata_function = build_parkour_hurdle  # 返回栏障路线 waypoint。
    terrain_class = 1  # 地形类别编号：栏障。

    hurdle_length_range: tuple[float, float] = (0.1, 0.4)  # 栏障沿 x 方向的长度。
    hurdle_height_min_range: tuple[float, float] = (0.1, 0.2)  # 栏障高度采样下界。
    hurdle_height_max_range: tuple[float, float] = (0.15, 0.4)  # 栏障高度采样上界。

    obstacle_spacing_range: tuple[float, float] = (1.2, 2.2)  # 相邻栏障中心的 x 距离。
    lateral_offset_range: tuple[float, float] = (-0.4, 0.4)  # 栏障和路线目标的 y 偏移。
    half_hurdle_width_range: tuple[float, float] = (0.4, 0.8)  # 栏障有效宽度的一半。


@configclass
class ParkourFlatTerrainCfg(ParkourHurdleTerrainCfg):
    """Offset waypoint route without hurdle geometry."""

    function = parkour_flat_terrain  # IsaacLab 标准平地路线入口。
    metadata_function = build_parkour_flat  # 返回带横向转向的平地 waypoint。
    terrain_class = 2  # 地形类别编号：平地转向。

    obstacle_spacing_range: tuple[float, float] = (1.5, 2.4)  # 平地路线相邻导航段的 x 推进距离。
    half_hurdle_width_range: tuple[float, float] = (0.45, 1.0)  # 保留兼容的半宽采样范围。

    hurdle_height_min_range: tuple[float, float] = (0.1, 0.2)  # flat=True 时不会创建栏障。
    hurdle_height_max_range: tuple[float, float] = (0.15, 0.3)  # 继承字段，平地模式下不使用。


@configclass
class ParkourStepTerrainCfg(ParkourBaseTerrainCfg):
    """Ascending and descending offset step platforms."""

    function = parkour_step_terrain  # IsaacLab 标准台阶地形入口。
    metadata_function = build_parkour_step  # 返回台阶顶部 waypoint。
    terrain_class = 3  # 地形类别编号：升降台阶。

    step_height_range: tuple[float, float] = (0.06, 0.18)  # 简单到困难的单级高度。
    step_length_min_range: tuple[float, float] = (1.20, 0.75)  # 踏面长度采样下界。
    step_length_max_range: tuple[float, float] = (1.50, 1.05)  # 踏面长度采样上界。
    step_half_width_range: tuple[float, float] = (0.90, 0.60)  # 可行走台阶区域的半宽。
    step_lateral_offset_max_range: tuple[float, float] = (0.00, 0.20)  # 台阶中心最大左右偏移。

    start_waypoint_margin: float = 0.50  # 起点目标距起点平台末端的距离。
    final_platform_min_length: float = 2.00  # 终点平地必须保留的最小长度。
    final_goal_inset: float = 1.00  # 最终目标距终点平台前缘的距离。


@configclass
class ParkourGapTerrainCfg(ParkourBaseTerrainCfg):
    """Repeated gaps connected by laterally offset valid corridors."""

    function = parkour_gap_terrain  # IsaacLab 标准沟壑地形入口。
    metadata_function = build_parkour_gap  # 返回安全走廊 waypoint。
    terrain_class = 4  # 地形类别编号：沟壑。

    gap_size_range: tuple[float, float] = (0.1, 0.8)  # 简单到困难的单个沟壑宽度。
    gap_depth_range: tuple[float, float] = (0.2, 1.0)  # 沟壑底部深度采样范围。
    safe_run_length_range: tuple[float, float] = (0.8, 1.5)  # 相邻沟壑边缘间的安全地面长度。

    lateral_offset_range: tuple[float, float] = (-0.4, 0.4)  # 安全走廊中心的 y 偏移。
    half_valid_width_range: tuple[float, float] = (0.6, 1.2)  # 安全走廊宽度的一半。


@configclass
class ExtremeParkourTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Select the metadata-aware Extreme Parkour generator."""

    class_type: type = ExtremeParkourTerrainGenerator  # 实际创建的自定义生成器类。


@configclass
class ExtremeParkourTerrainImporterCfg(TerrainImporterCfg):
    """Select the metadata-aware Extreme Parkour importer."""

    class_type: type = ExtremeParkourTerrainImporter  # 实际创建的自定义导入器类。


def make_extreme_parkour_training_terrain_cfg(
    *,
    num_rows: int = 10,
    num_cols: int = 40,
    seed: int = 0,
    difficulty_range: tuple[float, float] = (0.0, 1.0),
    color_scheme: Literal["height", "random", "none"] = "none",
) -> ExtremeParkourTerrainGeneratorCfg:
    """Create the five-family curriculum terrain configuration."""
    return ExtremeParkourTerrainGeneratorCfg(
        seed=seed,  # 全局地形随机种子。
        curriculum=True,  # 行方向按照难度课程生成。
        size=(18.0, 4.0),  # 单个 tile 的长度和宽度。
        border_width=5.0,  # 整张地形外围边界宽度。
        border_height=1.0,  # 外围边界向下延伸深度。
        num_rows=num_rows,  # 难度行数。
        num_cols=num_cols,  # 地形类型列数。
        horizontal_scale=0.05,  # 生成器全局水平分辨率。
        vertical_scale=0.005,  # 生成器全局垂直分辨率。
        slope_threshold=1.5,  # 高度场转网格时的坡度阈值，保留兼容。
        color_scheme=color_scheme,  # 顶点着色方式。
        difficulty_range=difficulty_range,  # 第一行到最后一行的难度范围。
        use_cache=False,  # waypoint 元数据未进入 IsaacLab 标准缓存。
        sub_terrains={
            # Dictionary order defines terrain_class and the column groups.
            "parkour": ParkourTerrainCfg(proportion=0.2),
            "parkour_hurdle": ParkourHurdleTerrainCfg(
                proportion=0.2
            ),
            "parkour_flat": ParkourFlatTerrainCfg(
                proportion=0.2
            ),
            "parkour_step": ParkourStepTerrainCfg(
                proportion=0.2
            ),
            "parkour_gap": ParkourGapTerrainCfg(
                proportion=0.2
            ),
        },
    )


EXTREME_PARKOUR_TRAINING_TERRAINS_CFG = make_extreme_parkour_training_terrain_cfg()
