"""Mesh construction for a continuous Extreme-Parkour-style course.

The public :func:`parkour_terrain` function follows Isaac Lab's
``SubTerrainBaseCfg.function`` contract exactly:

``function(difficulty, cfg) -> (list[trimesh.Trimesh], np.ndarray)``.

One generated *sub-terrain* is a complete course, not one isolated obstacle.
Isaac Lab's :class:`TerrainGenerator` is therefore still responsible for
replicating the course, arranging copies in rows/columns, centering the final
mesh, and converting the returned local spawn origin into world coordinates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh
import torch

from isaaclab.terrains import TerrainGenerator, TerrainImporter

if TYPE_CHECKING:
    from .parkour_terrain_cfg import ExtremeParkourTerrainCfg


def _lerp(value_range: tuple[float, float], difficulty: float) -> float:
    """Linearly interpolate a difficulty-dependent scalar."""
    return value_range[0] + difficulty * (value_range[1] - value_range[0])


def _box_with_top(
    *,
    x_start: float,
    length: float,
    center_y: float,
    width: float,
    top_z: float,
    bottom_z: float,
) -> trimesh.Trimesh:
    """Create a box by specifying its top and bottom surfaces.

    ``trimesh.creation.box`` expects full extents and a transform that locates
    the box *center*.  Terrain design is easier to read in terms of a desired
    top surface, so this helper performs the center conversion once.
    """
    if length <= 0.0 or width <= 0.0 or top_z <= bottom_z:
        raise ValueError(f"Invalid box dimensions: length={length}, width={width}, top_z={top_z}, bottom_z={bottom_z}.")
    extents = (length, width, top_z - bottom_z)
    center = (x_start + 0.5 * length, center_y, 0.5 * (top_z + bottom_z))
    transform = trimesh.transformations.translation_matrix(center)
    return trimesh.creation.box(extents=extents, transform=transform)


def _ramp_prism(
    *,
    x_start: float,
    length: float,
    center_y: float,
    width: float,
    start_z: float,
    end_z: float,
    bottom_z: float,
) -> trimesh.Trimesh:
    """Create a watertight prism whose top face slopes along +x.

    Rotating a thin box can leave one end below the neighboring platform and
    create collision seams.  This helper describes the top endpoints directly
    and keeps a horizontal foundation at ``bottom_z``.
    """
    if length <= 0.0 or width <= 0.0 or bottom_z >= min(start_z, end_z):
        raise ValueError("Ramp length/width must be positive and bottom_z must be below both top endpoints.")

    x0 = x_start
    x1 = x_start + length
    y0 = center_y - 0.5 * width
    y1 = center_y + 0.5 * width
    vertices = np.array(
        [
            [x0, y0, bottom_z],
            [x1, y0, bottom_z],
            [x1, y1, bottom_z],
            [x0, y1, bottom_z],
            [x0, y0, start_z],
            [x1, y0, end_z],
            [x1, y1, end_z],
            [x0, y1, start_z],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def build_parkour_course(difficulty: float, cfg: ExtremeParkourTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Build one complete course in the local ``[0, size]`` terrain frame.

    Course order along +x:

    1. start platform;
    2. hurdle on a continuous runway;
    3. two-step ascent and descent;
    4. open gap;
    5. separated, laterally offset stepping platforms;
    6. up/down ramp pair;
    7. finish platform.

    The geometry contains no full-length plate at ``ground_height`` because
    that would fill the gap and stepping-stone voids.  It does contain a
    finite recessed floor at ``ground_height - foundation_depth``.  This
    mirrors Extreme Parkour's finite-depth height-field pits without adding an
    infinite ground plane.
    """
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    course_length, cell_width = cfg.size
    expected_length = (
        cfg.start_length
        + cfg.hurdle_section_length
        + cfg.step_section_length
        + cfg.gap_section_length
        + cfg.stone_section_length
        + cfg.ramp_section_length
        + cfg.finish_length
    )
    if not np.isclose(expected_length, course_length, atol=1.0e-6):
        raise ValueError(
            "The parkour segment lengths must sum to cfg.size[0]: "
            f"segments={expected_length:.3f} m, cfg.size[0]={course_length:.3f} m."
        )
    if cfg.track_width > cell_width:
        raise ValueError(f"track_width={cfg.track_width} cannot exceed cfg.size[1]={cell_width}.")
    if cfg.num_stones < 2:
        raise ValueError("num_stones must be at least 2.")
    if cfg.foundation_depth <= 0.0:
        raise ValueError("foundation_depth must be positive.")
    if cfg.pit_floor_thickness <= 0.0:
        raise ValueError("pit_floor_thickness must be positive.")

    meshes: list[trimesh.Trimesh] = []
    center_y = 0.5 * cell_width
    ground_bottom = cfg.ground_height - cfg.foundation_depth
    cursor = 0.0

    # Finite pit floor.  Its top is below all normal walking surfaces, so the
    # gap and stepping-stone sections remain real drops.  It spans only this
    # sub-terrain cell; no infinite GroundPlane is created.
    meshes.append(
        _box_with_top(
            x_start=0.0,
            length=course_length,
            center_y=center_y,
            width=cell_width,
            top_z=ground_bottom,
            bottom_z=ground_bottom - cfg.pit_floor_thickness,
        )
    )

    # A small deterministic variation makes columns differ without changing
    # global NumPy state.  TerrainGenerator injects ``seed`` into the copied cfg.
    base_seed = 0 if getattr(cfg, "seed", None) is None else int(cfg.seed)  # type: ignore
    difficulty_key = int(round(difficulty * 1_000_000))
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, difficulty_key]))

    def add_track(x_start: float, length: float, top_z: float = cfg.ground_height) -> None:
        meshes.append(
            _box_with_top(
                x_start=x_start,
                length=length,
                center_y=center_y,
                width=cfg.track_width,
                top_z=top_z,
                bottom_z=ground_bottom,
            )
        )

    # 1) Start platform. The returned origin is near its beginning, not at the
    # geometric center of the entire 20 m course.
    add_track(cursor, cfg.start_length)
    origin = np.array([cfg.spawn_x, center_y, cfg.ground_height], dtype=np.float64)
    cursor += cfg.start_length

    # 2) Hurdle: runway remains continuous while a solid box rises above it.
    hurdle_start = cursor
    add_track(hurdle_start, cfg.hurdle_section_length)
    hurdle_height = _lerp(cfg.hurdle_height_range, difficulty)
    hurdle_x = hurdle_start + cfg.hurdle_runup
    meshes.append(
        _box_with_top(
            x_start=hurdle_x,
            length=cfg.hurdle_depth,
            center_y=center_y,
            width=cfg.track_width,
            top_z=cfg.ground_height + hurdle_height,
            bottom_z=cfg.ground_height,
        )
    )
    cursor += cfg.hurdle_section_length

    # 3) Step: approach -> +h -> +2h -> +h -> exit.  Each platform is a solid
    # prism down to the shared foundation so PhysX sees closed collision meshes.
    step_height = _lerp(cfg.step_height_range, difficulty)
    step_run = 0.5 * (cfg.step_section_length - 3.0 * cfg.step_tread_length)
    if step_run < 0.0:
        raise ValueError("step_section_length is too short for three step_tread_length values.")
    add_track(cursor, step_run)
    cursor += step_run
    for level in (1, 2, 1):
        add_track(cursor, cfg.step_tread_length, cfg.ground_height + level * step_height)
        cursor += cfg.step_tread_length
    add_track(cursor, step_run)
    cursor += step_run

    # 4) Gap: create the approach and landing surfaces but intentionally append
    # no mesh inside [gap_start, gap_end].
    gap_length = _lerp(cfg.gap_length_range, difficulty)
    gap_run = 0.5 * (cfg.gap_section_length - gap_length)
    if gap_run <= 0.0:
        raise ValueError("gap_section_length must be larger than the maximum generated gap.")
    add_track(cursor, gap_run)
    cursor += gap_run + gap_length
    add_track(cursor, gap_run)
    cursor += gap_run

    # 5) Stepping platforms: separated boxes over a void.  Increasing
    # difficulty shrinks their x/y footprint, increases lateral offsets and adds
    # small height changes.
    stone_length = _lerp(cfg.stone_length_range, difficulty)
    stone_width = _lerp(cfg.stone_width_range, difficulty)
    stone_height_variation = _lerp(cfg.stone_height_range, difficulty)
    lateral_offset = _lerp(cfg.stone_lateral_offset_range, difficulty)
    stone_pitch = cfg.stone_section_length / cfg.num_stones
    if stone_length >= stone_pitch:
        raise ValueError("stone_length must be smaller than stone_section_length / num_stones to preserve gaps.")
    for stone_index in range(cfg.num_stones):
        stone_center_x = cursor + (stone_index + 0.5) * stone_pitch
        side = -1.0 if stone_index % 2 == 0 else 1.0
        jitter = rng.uniform(-0.15, 0.15) * lateral_offset
        stone_center_y = center_y + side * lateral_offset + jitter
        stone_top = cfg.ground_height + rng.uniform(0.0, stone_height_variation)
        meshes.append(
            _box_with_top(
                x_start=stone_center_x - 0.5 * stone_length,
                length=stone_length,
                center_y=stone_center_y,
                width=stone_width,
                top_z=stone_top,
                bottom_z=ground_bottom,
            )
        )
    cursor += cfg.stone_section_length

    # 6) Ramp pair: an uphill prism followed immediately by a downhill prism.
    ramp_peak = _lerp(cfg.ramp_height_range, difficulty)
    half_ramp_length = 0.5 * cfg.ramp_section_length
    meshes.append(
        _ramp_prism(
            x_start=cursor,
            length=half_ramp_length,
            center_y=center_y,
            width=cfg.track_width,
            start_z=cfg.ground_height,
            end_z=cfg.ground_height + ramp_peak,
            bottom_z=ground_bottom,
        )
    )
    cursor += half_ramp_length
    meshes.append(
        _ramp_prism(
            x_start=cursor,
            length=half_ramp_length,
            center_y=center_y,
            width=cfg.track_width,
            start_z=cfg.ground_height + ramp_peak,
            end_z=cfg.ground_height,
            bottom_z=ground_bottom,
        )
    )
    cursor += half_ramp_length

    # 7) Finish platform.
    add_track(cursor, cfg.finish_length)
    cursor += cfg.finish_length
    if not np.isclose(cursor, course_length, atol=1.0e-6):
        raise RuntimeError(f"Internal course cursor ended at {cursor:.3f} m, expected {course_length:.3f} m.")

    return meshes, origin


def continuous_parkour_terrain(
    difficulty: float,
    cfg: ExtremeParkourTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Isaac Lab terrain-function entry point."""
    return build_parkour_course(difficulty, cfg)

# -----------------------------------------------------------------------------
# Extreme Parkour training terrains
# -----------------------------------------------------------------------------

def _rng_from_cfg(cfg) -> np.random.Generator:
    """Create the tile-local random-number generator.

    ExtremeParkourTerrainGenerator injects a deterministic seed into the copied
    sub-terrain config before calling the builder.
    """
    seed = getattr(cfg, "seed", 0)
    if seed is None:
        seed = 0
    return np.random.default_rng(int(seed))

def _quantize(value: float, resolution: float) -> float:
    """Round a metric value to a configured terrain resolution."""
    return round(value / resolution) * resolution

def _sample_grid(
    rng: np.random.Generator,
    lower: float,
    upper: float,
    resolution: float,
) -> float:
    """Replicate Isaac Gym's integer-pixel sampling in metric units.

    ``upper`` is exclusive, matching ``np.random.randint(lower, upper)``.
    """
    lower_index = int(round(lower / resolution))
    upper_index = int(round(upper / resolution))

    if upper_index <= lower_index:
        return resolution * lower_index

    return float(rng.integers(lower_index, upper_index)) * resolution

def _sample_quantized_uniform(
    rng: np.random.Generator,
    lower: float,
    upper: float,
    resolution: float,
) -> float:
    if upper <= lower:
        return _quantize(lower, resolution)
    return _quantize(float(rng.uniform(lower, upper)), resolution)

def _parkour_context(difficulty: float, cfg):
    """Validate common parameters and return shared tile quantities."""
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    tile_length, tile_width = cfg.size

    if cfg.num_waypoints != 8:
        raise ValueError(
            f"Extreme Parkour requires 8 waypoints, got {cfg.num_waypoints}."
        )
    if cfg.num_obstacles != cfg.num_waypoints - 2:
        raise ValueError(
            "num_obstacles must equal num_waypoints - 2: "
            f"{cfg.num_obstacles} != {cfg.num_waypoints - 2}."
        )
    if tile_length <= cfg.platform_length:
        raise ValueError(
            f"Tile length {tile_length} must exceed platform length {cfg.platform_length}."
        )
    if cfg.foundation_depth <= 0.0:
        raise ValueError("foundation_depth must be positive.")
    if cfg.pit_floor_thickness <= 0.0:
        raise ValueError("pit_floor_thickness must be positive.")

    center_y = 0.5 * tile_width
    # foundation_depth describes the vertical extent of normal walkable bodies.
    # A pit floor has its own bottom: pit_top - pit_floor_thickness.
    foundation_bottom = cfg.ground_height - cfg.foundation_depth

    origin = np.array(
        [cfg.spawn_x, center_y, cfg.ground_height],
        dtype=np.float64,
    )

    rng = _rng_from_cfg(cfg)

    return (
        difficulty,
        float(tile_length),
        float(tile_width),
        center_y,
        foundation_bottom,
        origin,
        rng,
    )

def _append_box_surface(
    meshes: list[trimesh.Trimesh],
    *,  # 强制后面的参数必须用 key=value 传递，避免位置参数混淆。
    tile_length: float,
    tile_width: float,
    x_start: float,
    x_end: float,
    center_y: float,
    width: float,
    top_z: float,
    bottom_z: float,
) -> None:
    """Append a clipped, watertight box representing one walkable surface."""
    clipped_x_start = max(0.0, x_start)
    clipped_x_end = min(tile_length, x_end)

    y_start = max(0.0, center_y - 0.5 * width)
    y_end = min(tile_width, center_y + 0.5 * width)

    length = clipped_x_end - clipped_x_start
    clipped_width = y_end - y_start

    if length <= 1.0e-6 or clipped_width <= 1.0e-6:
        return

    if top_z <= bottom_z:
        raise ValueError(
            f"Surface top {top_z} must be above bottom {bottom_z}."
        )

    meshes.append(
        _box_with_top(
            x_start=clipped_x_start,
            length=length,
            center_y=0.5 * (y_start + y_end),
            width=clipped_width,
            top_z=top_z,
            bottom_z=bottom_z,
        )
    )

def _lateral_ramp_prism(
    *,
    x_start: float,
    length: float,
    center_y: float,
    width: float,
    left_top_z: float,
    right_top_z: float,
    bottom_z: float,
) -> trimesh.Trimesh:
    """Create a closed platform whose top is inclined along the y-axis."""
    if length <= 0.0 or width <= 0.0:
        raise ValueError("Ramp length and width must be positive.")
    if bottom_z >= min(left_top_z, right_top_z):
        raise ValueError(
            "bottom_z must be lower than both lateral top endpoints."
        )

    x0 = x_start
    x1 = x_start + length
    y0 = center_y - 0.5 * width
    y1 = center_y + 0.5 * width

    vertices = np.array(
        [
            [x0, y0, bottom_z],
            [x1, y0, bottom_z],
            [x1, y1, bottom_z],
            [x0, y1, bottom_z],
            [x0, y0, left_top_z],
            [x1, y0, left_top_z],
            [x1, y1, right_top_z],
            [x0, y1, right_top_z],
        ],
        dtype=np.float64,
    )

    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )


def build_parkour(
    difficulty: float,
    cfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray]:
    """Build the laterally alternating inclined-platform terrain."""
    (
        difficulty,
        tile_length,
        tile_width,
        center_y,
        foundation_bottom,
        origin,
        rng,
    ) = _parkour_context(difficulty, cfg)

    meshes: list[trimesh.Trimesh] = []
    waypoints = np.zeros((cfg.num_waypoints, 3), dtype=np.float64)

    pit_depth = _sample_quantized_uniform(
        rng,
        cfg.pit_depth_range[0],
        cfg.pit_depth_range[1],
        cfg.vertical_scale,
    )

    pit_top = cfg.ground_height - pit_depth

    # Finite pit floor.
    _append_box_surface(
        meshes,
        tile_length=tile_length,
        tile_width=tile_width,
        x_start=0.0,
        x_end=tile_length,
        center_y=center_y,
        width=tile_width,
        top_z=pit_top,
        bottom_z=pit_top - cfg.pit_floor_thickness,
    )

    # Starting platform.
    _append_box_surface(
        meshes,
        tile_length=tile_length,
        tile_width=tile_width,
        x_start=0.0,
        x_end=cfg.platform_length,
        center_y=center_y,
        width=tile_width,
        top_z=cfg.ground_height,
        bottom_z=foundation_bottom,
    )

    stone_length_lower = _lerp(cfg.stone_length_lower_range, difficulty)
    stone_length_upper = _lerp(cfg.stone_length_upper_range, difficulty)

    stone_length = float(rng.uniform(stone_length_lower, stone_length_upper))
    stone_length = 2.0 * round(stone_length / 2.0, 1)

    gap_max = _lerp(cfg.stone_gap_max_range, difficulty)
    dis_x_min = stone_length + cfg.stone_gap_min
    dis_x_max = stone_length + gap_max

    lateral_max = _lerp(cfg.lateral_offset_max_range, difficulty)
    incline_height = _quantize(
        _lerp(cfg.incline_height_range, difficulty),
        cfg.vertical_scale,
    )

    last_incline_height = _quantize(
        _lerp(cfg.last_incline_height_range, difficulty),
        cfg.vertical_scale,
    )

    first_spacing = _sample_grid(
        rng,
        dis_x_min,
        dis_x_max,
        cfg.horizontal_scale,
    )

    stone_center_x = (
        cfg.platform_length
        - first_spacing
        + stone_length * 0.5
    )

    waypoints[0] = np.array(
        [
            cfg.platform_length - 0.5 * stone_length,
            center_y,
            cfg.ground_height,
        ]
    )

    left_right_flag = int(rng.integers(0, 2))

    for obstacle_index in range(cfg.num_obstacles):
        spacing = _sample_grid(
            rng,
            dis_x_min,
            dis_x_max,
            cfg.horizontal_scale,
        )
        stone_center_x += spacing

        side_sign = 1.0 if left_right_flag == 1 else -1.0
        lateral_offset = _sample_grid(
            rng,
            cfg.lateral_offset_min,
            lateral_max,
            cfg.horizontal_scale,
        )
        stone_center_y = center_y + side_sign * lateral_offset

        is_last_stone = obstacle_index == cfg.num_obstacles - 1
        if is_last_stone:
            current_length = cfg.last_stone_length
            current_incline = last_incline_height
            stone_center_x += 0.25 * cfg.last_stone_length
        else:
            current_length = stone_length
            current_incline = incline_height

        x_start = stone_center_x - 0.5 * current_length

        if x_start < tile_length and x_start + current_length > 0.0:
            meshes.append(
                _lateral_ramp_prism(
                    x_start=x_start,
                    length=current_length,
                    center_y=stone_center_y,
                    width=cfg.stone_width,
                    left_top_z=(
                        cfg.ground_height
                        - side_sign * current_incline
                    ),
                    right_top_z=(
                        cfg.ground_height
                        + side_sign * current_incline
                    ),
                    bottom_z=foundation_bottom,
                )
            )

        # At the center of the symmetric lateral incline, z is ground_height.
        waypoints[obstacle_index + 1] = np.array(
            [
                stone_center_x,
                stone_center_y,
                cfg.ground_height,
            ]
        )

        left_right_flag = 1 - left_right_flag


    final_spacing = 2.0 * _sample_grid(
        rng,
        dis_x_min,
        dis_x_max,
        cfg.horizontal_scale,
    )

    final_goal_x = min(
        stone_center_x + final_spacing,
        tile_length - cfg.finish_goal_margin,
    )

    last_stone_right_edge = stone_center_x + 0.5 * cfg.last_stone_length

    final_platform_start = last_stone_right_edge + cfg.final_platform_gap

    final_goal_max_x = tile_length - cfg.finish_goal_margin
    if final_platform_start > final_goal_max_x:
        raise ValueError(
            "The last stone leaves no usable space for the final platform: "
            f"final_platform_start={final_platform_start:.3f}, "
            f"final_goal_max_x={final_goal_max_x:.3f}, "
            f"tile_length={tile_length:.3f}."
        )

    _append_box_surface(
        meshes,
        tile_length=tile_length,
        tile_width=tile_width,
        x_start=final_platform_start,
        x_end=tile_length,
        center_y=center_y,
        width=tile_width,
        top_z=cfg.ground_height,
        bottom_z=foundation_bottom,
    )

    final_goal_x = float(
        np.clip(
            final_goal_x,
            final_platform_start,
            final_goal_max_x,
        )
    )
    waypoints[-1] = np.array(
        [final_goal_x, center_y, cfg.ground_height]
    )

    return meshes, origin, waypoints

def _build_hurdle_or_flat(
    difficulty: float,
    cfg,
    *,
    flat: bool,
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray]:
    """Shared implementation for hurdle and flat variants."""
    (
        difficulty,
        tile_length,
        tile_width,
        center_y,
        foundation_bottom,
        origin,
        rng,
    ) = _parkour_context(difficulty, cfg)

    meshes: list[trimesh.Trimesh] = []
    waypoints = np.zeros((cfg.num_waypoints, 3), dtype=np.float64)

    # Full finite ground slab.
    _append_box_surface(
        meshes,
        tile_length=tile_length,
        tile_width=tile_width,
        x_start=0.0,
        x_end=tile_length,
        center_y=center_y,
        width=tile_width,
        top_z=cfg.ground_height,
        bottom_z=foundation_bottom,
    )

    hurdle_length = _quantize(
        _lerp(cfg.hurdle_length_range, difficulty),
        cfg.horizontal_scale,
    )

    half_hurdle_width = _sample_quantized_uniform(
        rng,
        cfg.half_hurdle_width_range[0],
        cfg.half_hurdle_width_range[1],
        cfg.horizontal_scale,
    )

    hurdle_height_min = _lerp(
        cfg.hurdle_height_min_range,
        difficulty,
    )

    hurdle_height_max = _lerp(
        cfg.hurdle_height_max_range,
        difficulty,
    )

    cursor_x = cfg.platform_length

    # Keep the first waypoint one horizontal grid cell before the platform edge.
    waypoints[0] = np.array(
        [
            cfg.platform_length - cfg.horizontal_scale,
            center_y,
            cfg.ground_height,
        ]
    )

    for obstacle_index in range(cfg.num_obstacles):
        spacing = _sample_grid(
            rng,
            cfg.obstacle_spacing_range[0],
            cfg.obstacle_spacing_range[1],
            cfg.horizontal_scale,
        )
        lateral_offset = _sample_grid(
            rng,
            cfg.lateral_offset_range[0],
            cfg.lateral_offset_range[1],
            cfg.horizontal_scale,
        )

        cursor_x += spacing
        obstacle_center_y = center_y + lateral_offset

        if not flat:
            hurdle_height = _sample_grid(
                rng,
                hurdle_height_min,
                hurdle_height_max,
                cfg.vertical_scale,
            )

            hurdle_height = max(
                hurdle_height,
                cfg.vertical_scale,
            )

            _append_box_surface(
                meshes,
                tile_length=tile_length,
                tile_width=tile_width,
                x_start=cursor_x - 0.5 * hurdle_length,
                x_end=cursor_x + 0.5 * hurdle_length,
                center_y=obstacle_center_y,
                width=2.0 * half_hurdle_width,
                top_z=cfg.ground_height + hurdle_height,
                bottom_z=cfg.ground_height,
            )

        waypoints[obstacle_index + 1] = np.array(
            [
                cursor_x - 0.5 * spacing,
                obstacle_center_y,
                cfg.ground_height,
            ]
        )

    final_spacing = _sample_grid(
        rng,
        cfg.obstacle_spacing_range[0],
        cfg.obstacle_spacing_range[1],
        cfg.horizontal_scale,
    )

    final_goal_x = min(
        cursor_x + final_spacing,
        tile_length - cfg.finish_goal_margin,
    )

    waypoints[-1] = np.array(
        [final_goal_x, center_y, cfg.ground_height]
    )

    return meshes, origin, waypoints

def build_parkour_hurdle(
    difficulty: float,
    cfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray]:
    return _build_hurdle_or_flat(
        difficulty,
        cfg,
        flat=False,
    )

def build_parkour_flat(
    difficulty: float,
    cfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray]:
    return _build_hurdle_or_flat(
        difficulty,
        cfg,
        flat=True,
    )

def build_parkour_step(
    difficulty: float,
    cfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray]:
    """Build an ascending-then-descending step terrain."""
    (
        difficulty,
        tile_length,
        tile_width,
        center_y,
        foundation_bottom,
        origin,
        rng,
    ) = _parkour_context(difficulty, cfg)

    meshes: list[trimesh.Trimesh] = []
    waypoints = np.zeros((cfg.num_waypoints, 3), dtype=np.float64)

    _append_box_surface(
        meshes,
        tile_length=tile_length,
        tile_width=tile_width,
        x_start=0.0,
        x_end=tile_length,
        center_y=center_y,
        width=tile_width,
        top_z=cfg.ground_height,
        bottom_z=foundation_bottom,
    )

    step_height = _quantize(
        _lerp(cfg.step_height_range, difficulty),
        cfg.vertical_scale,
    )

    step_length_min = _quantize(
        _lerp(
            cfg.step_length_min_range,
            difficulty
        ),
        cfg.horizontal_scale,
    )

    step_length_max = _quantize(
        _lerp(
            cfg.step_length_max_range,
            difficulty
        ),
        cfg.horizontal_scale,
    )

    if step_length_min <= 0.0:
        raise ValueError(
            "step_length_min must be positive."
        )

    if step_length_max < step_length_min:
        raise ValueError(
            "step_length_max must not be smaller "
            "than step_length_min."
        )

    step_half_width = _quantize(
        _lerp(cfg.step_half_width_range, difficulty),
        cfg.horizontal_scale,
    )

    lateral_offset_max = _quantize(
        _lerp(
            cfg.step_lateral_offset_max_range,
            difficulty,
        ),
        cfg.horizontal_scale,
    )

    if step_half_width + lateral_offset_max > 0.5 * tile_width:
        raise ValueError(
            "The step corridor may exceed the terrain width: "
            f"half_width={step_half_width:.3f}, "
            f"lateral_offset_max={lateral_offset_max:.3f}, "
            f"tile_width={tile_width:.3f}."
        )

    maximum_required_length = (
        cfg.platform_length
        + cfg.num_obstacles * step_length_max
        + cfg.final_platform_min_length
    )

    if maximum_required_length > tile_length:
        raise ValueError(
            "The configured steps cannot fit inside the tile: "
            f"required={maximum_required_length:.3f}, "
            f"tile_length={tile_length:.3f}."
        )

    # 起点 waypoint。
    waypoints[0] = np.array(
        [
            cfg.platform_length - cfg.start_waypoint_margin,
            center_y,
            cfg.ground_height,
        ],
        dtype=np.float64,
    )

    cursor_x = cfg.platform_length

    for step_index in range(cfg.num_obstacles):
        current_step_length = _sample_grid(
            rng,
            step_length_min,
            step_length_max,
            cfg.horizontal_scale,
        )

        height_level = min(
            step_index + 1,
            cfg.num_obstacles - step_index,
        )

        current_top_z = (
            cfg.ground_height
            + height_level * step_height
        )

        lateral_offset = _sample_grid(
            rng,
            -lateral_offset_max,
            lateral_offset_max + cfg.horizontal_scale,
            cfg.horizontal_scale,
        )

        current_center_y = (
            center_y + lateral_offset
        )

        x_start = cursor_x
        x_end = cursor_x + current_step_length

        _append_box_surface(
            meshes,
            tile_length=tile_length,
            tile_width=tile_width,
            x_start=x_start,
            x_end=x_end,
            center_y=current_center_y,
            width=2.0 * step_half_width,
            top_z=current_top_z,
            bottom_z=cfg.ground_height,
        )

        waypoints[step_index + 1] = np.array(
            [
                x_start + 0.5 * current_step_length,
                current_center_y,
                current_top_z,
            ],
            dtype=np.float64,
        )

        cursor_x = x_end

    # cursor_x 后面的底层平地就是最终平台。
    final_platform_length = (
        tile_length - cursor_x
    )

    if final_platform_length < cfg.final_platform_min_length:
        raise ValueError(
            "The generated final platform is too short: "
            f"length={final_platform_length:.3f}, "
            f"required={cfg.final_platform_min_length:.3f}."
        )

    final_goal_min_x = (
        cursor_x + cfg.final_goal_inset
    )

    final_goal_max_x = (
        tile_length - cfg.finish_goal_margin
    )

    if final_goal_min_x > final_goal_max_x:
        raise ValueError(
            "There is no valid position for the final waypoint: "
            f"minimum={final_goal_min_x:.3f}, "
            f"maximum={final_goal_max_x:.3f}."
        )

    waypoints[-1] = np.array(
        [
            final_goal_min_x,
            center_y,
            cfg.ground_height,
        ],
        dtype=np.float64,
    )

    return meshes, origin, waypoints

def build_parkour_gap(
    difficulty: float,
    cfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray, np.ndarray]:
    """Build repeated full-width gaps connected by offset safe corridors."""
    (
        difficulty,
        tile_length,
        tile_width,
        center_y,
        foundation_bottom,
        origin,
        rng,
    ) = _parkour_context(difficulty, cfg)

    meshes: list[trimesh.Trimesh] = []
    waypoints = np.zeros((cfg.num_waypoints, 3), dtype=np.float64)

    gap_size = _quantize(
        _lerp(cfg.gap_size_range, difficulty),
        cfg.horizontal_scale,
    )
    pit_depth = _sample_quantized_uniform(
        rng,
        cfg.gap_depth_range[0],
        cfg.gap_depth_range[1],
        cfg.vertical_scale,
    )
    pit_top = cfg.ground_height - pit_depth

    half_valid_width = _sample_quantized_uniform(
        rng,
        cfg.half_valid_width_range[0],
        cfg.half_valid_width_range[1],
        cfg.horizontal_scale,
    )

    # Finite floor at the bottom of all voids.
    _append_box_surface(
        meshes,
        tile_length=tile_length,
        tile_width=tile_width,
        x_start=0.0,
        x_end=tile_length,
        center_y=center_y,
        width=tile_width,
        top_z=pit_top,
        bottom_z=pit_top - cfg.pit_floor_thickness,
    )

    # Full-width starting platform.
    _append_box_surface(
        meshes,
        tile_length=tile_length,
        tile_width=tile_width,
        x_start=0.0,
        x_end=cfg.platform_length,
        center_y=center_y,
        width=tile_width,
        top_z=cfg.ground_height,
        bottom_z=foundation_bottom,
    )

    center_spacing_min = cfg.safe_run_length_range[0] + gap_size
    center_spacing_max = cfg.safe_run_length_range[1] + gap_size

    cursor_x = cfg.platform_length
    safe_segment_start = cfg.platform_length

    waypoints[0] = np.array(
        [
            cfg.platform_length - cfg.horizontal_scale,
            center_y,
            cfg.ground_height,
        ]
    )

    for obstacle_index in range(cfg.num_obstacles):
        center_spacing = _sample_grid(
            rng,
            center_spacing_min,
            center_spacing_max,
            cfg.horizontal_scale,
        )
        lateral_offset = _sample_grid(
            rng,
            cfg.lateral_offset_range[0],
            cfg.lateral_offset_range[1],
            cfg.horizontal_scale,
        )

        cursor_x += center_spacing
        corridor_center_y = center_y + lateral_offset

        gap_start = cursor_x - 0.5 * gap_size
        gap_end = gap_start + gap_size

        _append_box_surface(
            meshes,
            tile_length=tile_length,
            tile_width=tile_width,
            x_start=safe_segment_start,
            x_end=gap_start,
            center_y=corridor_center_y,
            width=2.0 * half_valid_width,
            top_z=cfg.ground_height,
            bottom_z=foundation_bottom,
        )

        safe_segment_start = gap_end
        waypoints[obstacle_index + 1] = np.array(
            [
                cursor_x - 0.5 * center_spacing,
                corridor_center_y,
                cfg.ground_height,
            ]
        )

    if safe_segment_start > tile_length - cfg.finish_goal_margin:
        raise ValueError(
            "The final gap leaves no usable finish platform: "
            f"final_platform_start={safe_segment_start:.3f}, "
            f"tile_length={tile_length:.3f}, "
            f"finish_goal_margin={cfg.finish_goal_margin:.3f}."
        )

    _append_box_surface(
        meshes,
        tile_length=tile_length,
        tile_width=tile_width,
        x_start=safe_segment_start,
        x_end=tile_length,
        center_y=center_y,
        width=tile_width,
        top_z=cfg.ground_height,
        bottom_z=foundation_bottom,
    )

    final_spacing = _sample_grid(
        rng,
        center_spacing_min,
        center_spacing_max,
        cfg.horizontal_scale,
    )

    final_goal_x = float(
        np.clip(
            cursor_x + final_spacing,
            safe_segment_start,
            tile_length - cfg.finish_goal_margin,
        )
    )

    waypoints[-1] = np.array(
        [final_goal_x, center_y, cfg.ground_height]
    )

    return meshes, origin, waypoints

# -----------------------------------------------------------------------------
# Standard IsaacLab SubTerrainBaseCfg.function wrappers
# -----------------------------------------------------------------------------


def parkour_terrain(
    difficulty: float,
    cfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    meshes, origin, _ = build_parkour(difficulty, cfg)
    return meshes, origin

def parkour_hurdle_terrain(
    difficulty: float,
    cfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    meshes, origin, _ = build_parkour_hurdle(difficulty, cfg)
    return meshes, origin

def parkour_flat_terrain(
    difficulty: float,
    cfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    meshes, origin, _ = build_parkour_flat(difficulty, cfg)
    return meshes, origin

def parkour_step_terrain(
    difficulty: float,
    cfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    meshes, origin, _ = build_parkour_step(difficulty, cfg)
    return meshes, origin

def parkour_gap_terrain(
    difficulty: float,
    cfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    meshes, origin, _ = build_parkour_gap(difficulty, cfg)
    return meshes, origin


# -----------------------------------------------------------------------------
# Generator with waypoint/class/difficulty metadata
# -----------------------------------------------------------------------------


class ExtremeParkourTerrainGenerator(TerrainGenerator):
    """TerrainGenerator with waypoint, class, and difficulty metadata."""

    def __init__(self, cfg, device: str = "cpu"):
        if not cfg.curriculum:
            raise ValueError(
                "Extreme Parkour training terrain requires curriculum=True."
            )
        if cfg.use_cache:
            raise ValueError(
                "use_cache must be False because IsaacLab's mesh cache does not "
                "store waypoint/class/difficulty metadata."
            )
        if len(cfg.sub_terrains) != 5:
            raise ValueError(
                "Extreme Parkour requires exactly five sub-terrain types."
            )

        waypoint_counts = {
            int(sub_cfg.num_waypoints)  # type: ignore
            for sub_cfg in cfg.sub_terrains.values()
        }
        if waypoint_counts != {8}:
            raise ValueError(
                f"All sub-terrain types must have 8 waypoints, got {waypoint_counts}."
            )

        self.waypoints_grid = np.zeros(
            (cfg.num_rows, cfg.num_cols, 8, 3),
            dtype=np.float32,
        )
        self.terrain_class_grid = np.zeros(
            (cfg.num_rows, cfg.num_cols),
            dtype=np.int64,
        )
        self.difficulty_grid = np.zeros(
            (cfg.num_rows, cfg.num_cols),
            dtype=np.float32,
        )

        self.terrain_class_names = tuple(cfg.sub_terrains.keys())
        self._base_seed = 0 if cfg.seed is None else int(cfg.seed)

        # Parent initialization calls our overridden
        # _generate_curriculum_terrains().
        super().__init__(cfg=cfg, device=device)
        # TerrainGenerator centers the final terrain after generating all tiles.
        global_offset = np.array(
            [
                -0.5 * cfg.num_rows * cfg.size[0],
                -0.5 * cfg.num_cols * cfg.size[1],
                0.0,
            ],
            dtype=np.float32,
        )
        self.waypoints_grid += global_offset

        # TerrainImporter normally discards the generator instance. Temporarily
        # attach metadata to the generator config so our importer can copy it.
        self.cfg._runtime_waypoints_grid = self.waypoints_grid  # type: ignore
        self.cfg._runtime_terrain_class_grid = self.terrain_class_grid  # type: ignore
        self.cfg._runtime_difficulty_grid = self.difficulty_grid  # type: ignore
        self.cfg._runtime_terrain_class_names = self.terrain_class_names  # type: ignore

    def _column_sub_terrain_indices(self) -> np.ndarray:
        """Map each curriculum column to a sub-terrain config."""
        proportions = np.array(
            [
                float(sub_cfg.proportion)
                for sub_cfg in self.cfg.sub_terrains.values()
            ],
            dtype=np.float64,
        )
        proportions /= proportions.sum()
        cumulative = np.cumsum(proportions)

        indices = np.zeros(self.cfg.num_cols, dtype=np.int64)
        for column in range(self.cfg.num_cols):
            choice = column / self.cfg.num_cols + 0.001
            matches = np.nonzero(choice < cumulative)[0]
            if len(matches) == 0:
                indices[column] = len(proportions) - 1
            else:
                indices[column] = int(matches[0])

        return indices

    def _tile_seed(
        self,
        row: int,
        column: int,
        terrain_class: int,
    ) -> int:
        seed_sequence = np.random.SeedSequence(
            [
                self._base_seed,
                row,
                column,
                terrain_class,
            ]
        )
        return int(seed_sequence.generate_state(1, dtype=np.uint32,)[0])

    def _build_and_add_tile(
        self,
        *,
        row: int,
        column: int,
        difficulty: float,
        sub_terrain_cfg,
    ) -> None:
        """Build one local tile and place its geometry and metadata in the grid."""

        tile_cfg = sub_terrain_cfg.copy()
        tile_cfg.size = self.cfg.size

        terrain_class = int(tile_cfg.terrain_class)
        tile_cfg.seed = self._tile_seed(
            row,
            column,
            terrain_class,
        )

        metadata_function = tile_cfg.metadata_function
        meshes, local_origin, local_waypoints = metadata_function(
            difficulty,
            tile_cfg,
        )

        if len(meshes) == 0:
            raise RuntimeError(
                f"Terrain tile ({row}, {column}) returned no meshes."
            )
        if local_waypoints.shape != (8, 3):
            raise RuntimeError(
                "Expected local waypoint shape (8, 3), got "
                f"{local_waypoints.shape} at tile ({row}, {column})."
            )
        if not np.isfinite(local_waypoints).all():
            raise RuntimeError(
                f"Non-finite waypoints at tile ({row}, {column})."
            )
        if not all(mesh.is_watertight for mesh in meshes):
            raise RuntimeError(
                f"Non-watertight component at tile ({row}, {column})."
            )
        mesh = trimesh.util.concatenate(meshes)

        # The custom builders use the tile's lower-left corner as (0, 0).
        # IsaacLab expects each tile to be centered before grid placement.
        center_transform = np.eye(4)
        center_transform[0, 3] = -0.5 * self.cfg.size[0]
        center_transform[1, 3] = -0.5 * self.cfg.size[1]
        mesh.apply_transform(center_transform)

        centered_origin = (
            np.asarray(local_origin, dtype=np.float64)
            + center_transform[:3, 3]
        )

        centered_waypoints = (
            np.asarray(local_waypoints, dtype=np.float64)
            + center_transform[:3, 3]
        )

        # Parent helper moves the centered tile to its grid cell and records
        # terrain_origins[row, column].
        self._add_sub_terrain(
            mesh,
            centered_origin,
            row,
            column,
            tile_cfg,
        )

        # Apply the same grid-cell translation to the waypoint metadata.
        tile_translation = np.array(
            [
                (row + 0.5) * self.cfg.size[0],
                (column + 0.5) * self.cfg.size[1],
                0.0,
            ],
            dtype=np.float64,
        )
        self.waypoints_grid[row, column] = (
            centered_waypoints + tile_translation
        ).astype(np.float32)
        self.terrain_class_grid[row, column] = terrain_class
        self.difficulty_grid[row, column] = float(difficulty)

    def _generate_curriculum_terrains(self) -> None:
        column_indices = self._column_sub_terrain_indices()
        sub_terrain_cfgs = list(self.cfg.sub_terrains.values())

        lower, upper = self.cfg.difficulty_range
        for column in range(self.cfg.num_cols):
            sub_terrain_cfg = sub_terrain_cfgs[column_indices[column]]
            for row in range(self.cfg.num_rows):
                if self.cfg.num_rows == 1:
                    normalized_difficulty = 0.0
                else:
                    normalized_difficulty = row / (self.cfg.num_rows - 1)

                difficulty = lower + (upper - lower) * normalized_difficulty

                self._build_and_add_tile(
                    row=row,
                    column=column,
                    difficulty=float(difficulty),
                    sub_terrain_cfg=sub_terrain_cfg,
                )

'''
创建 ExtremeParkourTerrainImporter
        ↓
ExtremeParkourTerrainImporter.__init__()
        ↓
super().__init__(cfg)
        ↓
TerrainImporter.__init__()
        ↓
生成 terrain mesh
        ↓
configure_env_origins(terrain_generator.terrain_origins)
        ↓
_compute_env_origins_curriculum(...)
        ↓
创建 self.terrain_levels
'''
class ExtremeParkourTerrainImporter(TerrainImporter):
    """TerrainImporter that keeps the generator's route metadata."""

    def __init__(self, cfg):
        super().__init__(cfg)

        generator_cfg = cfg.terrain_generator
        if generator_cfg is None:
            raise ValueError(
                "ExtremeParkourTerrainImporter requires terrain_generator."
            )

        runtime_attributes = (
            "_runtime_waypoints_grid",
            "_runtime_terrain_class_grid",
            "_runtime_difficulty_grid",
            "_runtime_terrain_class_names",
        )
        missing = [
            name
            for name in runtime_attributes
            if not hasattr(generator_cfg, name)
        ]
        if missing:
            raise RuntimeError(
                "Extreme Parkour generator metadata was not produced: "
                f"{missing}."
            )

        self.waypoints_grid = torch.as_tensor(
            np.asarray(
                generator_cfg._runtime_waypoints_grid  # type: ignore
            ).copy(),
            dtype=torch.float32,
            device=self.device,
        )
        self.terrain_class_grid = torch.as_tensor(
            np.asarray(
                generator_cfg._runtime_terrain_class_grid  # type: ignore
            ).copy(),
            dtype=torch.long,
            device=self.device,
        )
        self.difficulty_grid = torch.as_tensor(
            np.asarray(
                generator_cfg._runtime_difficulty_grid  # type: ignore
            ).copy(),
            dtype=torch.float32,
            device=self.device,
        )
        self.terrain_class_names = tuple(
            generator_cfg._runtime_terrain_class_names  # type: ignore
        )

        # Do not keep large runtime arrays inside the config object.
        for attribute_name in runtime_attributes:
            delattr(generator_cfg, attribute_name)
