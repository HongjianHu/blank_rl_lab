"""Preview continuous-demo or five-family Extreme Parkour terrains."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Preview Blank RL Lab parkour terrains.")
parser.add_argument(
    "--terrain_mode",
    choices=("continuous", "training"),
    default="continuous",
    help="Select the mixed demo or five-family training terrain.",
)
parser.add_argument(
    "--difficulty",
    type=float,
    default=0.5,
    help="Fixed continuous-demo difficulty when curriculum is disabled.",
)
parser.add_argument(
    "--curriculum",
    action="store_true",
    help="Use curriculum rows for the continuous demo.",
)
parser.add_argument(
    "--num_rows",
    type=int,
    default=None,
    help="Override row count. Defaults: continuous=1, training=10.",
)
parser.add_argument(
    "--num_cols",
    type=int,
    default=None,
    help="Override column count. Defaults: continuous=1, training=40.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Deterministic terrain seed.",
)
parser.add_argument(
    "--show_origins",
    action="store_true",
    help="Show each tile's robot spawn origin.",
)
parser.add_argument(
    "--show_waypoints",
    action="store_true",
    help="Show all training-route waypoints as spheres.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=-1,
    help="Stop after N simulation steps; negative keeps the viewer running.",
)
parser.add_argument(
    "--color_scheme",
    choices=("height", "random", "none"),
    default="none",
    help="TerrainGenerator vertex-color mode.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

from blank_rl_lab.terrains.parkour import (
    ExtremeParkourTerrainImporterCfg,
    make_extreme_parkour_training_terrain_cfg,
    make_parkour_terrain_cfg,
)


def resolve_grid_shape() -> tuple[int, int]:
    """Resolve mode-specific defaults without changing command-line overrides."""
    if args_cli.terrain_mode == "training":
        default_rows = 10
        default_cols = 40
    else:
        default_rows = 1
        default_cols = 1

    num_rows = (
        default_rows
        if args_cli.num_rows is None
        else args_cli.num_rows
    )
    num_cols = (
        default_cols
        if args_cli.num_cols is None
        else args_cli.num_cols
    )

    if num_rows <= 0 or num_cols <= 0:
        raise ValueError("num_rows and num_cols must be positive.")

    return num_rows, num_cols


def design_scene() -> tuple[TerrainImporter, int, int]:
    """Create lighting and import the selected finite terrain mesh."""
    light_cfg = sim_utils.DomeLightCfg(
        intensity=1800.0,
        color=(0.8, 0.8, 0.8),
    )
    light_cfg.func("/World/Light", light_cfg)

    num_rows, num_cols = resolve_grid_shape()

    if args_cli.terrain_mode == "training":
        generator_cfg = make_extreme_parkour_training_terrain_cfg(
            num_rows=num_rows,
            num_cols=num_cols,
            seed=args_cli.seed,
            difficulty_range=(0.0, 1.0),
            color_scheme=args_cli.color_scheme,
        )
        importer_cfg_class = ExtremeParkourTerrainImporterCfg
    else:
        if args_cli.curriculum:
            difficulty_range = (0.0, 1.0)
        else:
            fixed_difficulty = float(
                min(max(args_cli.difficulty, 0.0), 1.0)
            )
            difficulty_range = (
                fixed_difficulty,
                fixed_difficulty,
            )

        generator_cfg = make_parkour_terrain_cfg(
            num_rows=num_rows,
            num_cols=num_cols,
            curriculum=args_cli.curriculum,
            difficulty_range=difficulty_range,
            seed=args_cli.seed,
            color_scheme=args_cli.color_scheme,
        )
        importer_cfg_class = TerrainImporterCfg

    importer_cfg = importer_cfg_class(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=generator_cfg,
        num_envs=num_rows * num_cols,
        max_init_terrain_level=0,
        debug_vis=args_cli.show_origins,
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
    )

    if args_cli.color_scheme != "none":
        importer_cfg.visual_material = None

    # 配置中的 class_type 决定创建标准 importer 还是带 waypoint 元数据的 importer。
    terrain = importer_cfg.class_type(importer_cfg)

    return terrain, num_rows, num_cols


def validate_parkour_metadata(
    terrain: TerrainImporter,
    num_rows: int,
    num_cols: int,
) -> None:
    """Validate metadata shapes, endpoints and the full 10x40 distribution."""
    expected_waypoint_shape = (
        num_rows,
        num_cols,
        8,
        3,
    )
    expected_grid_shape = (
        num_rows,
        num_cols,
    )

    if tuple(terrain.waypoints_grid.shape) != expected_waypoint_shape:  # type: ignore
        raise RuntimeError(
            "Invalid waypoint shape: "
            f"{tuple(terrain.waypoints_grid.shape)} != "  # type: ignore
            f"{expected_waypoint_shape}."
        )
    if tuple(terrain.terrain_class_grid.shape) != expected_grid_shape:  # type: ignore
        raise RuntimeError(
            "Invalid terrain class shape: "
            f"{tuple(terrain.terrain_class_grid.shape)}."  # type: ignore
        )
    if tuple(terrain.difficulty_grid.shape) != expected_grid_shape:  # type: ignore
        raise RuntimeError(
            "Invalid difficulty shape: "
            f"{tuple(terrain.difficulty_grid.shape)} != "  # type: ignore
            f"{expected_grid_shape}."
        )
    if not torch.isfinite(terrain.waypoints_grid).all():  # type: ignore
        raise RuntimeError("waypoints_grid contains NaN or Inf.")
    if not torch.isfinite(terrain.difficulty_grid).all():  # type: ignore
        raise RuntimeError("difficulty_grid contains NaN or Inf.")

    if num_rows > 1:
        expected_first = torch.zeros(
            num_cols,
            device=terrain.difficulty_grid.device,  # type: ignore
        )
        expected_last = torch.ones(
            num_cols,
            device=terrain.difficulty_grid.device,  # type: ignore
        )

        if not torch.allclose(
            terrain.difficulty_grid[0],  # type: ignore
            expected_first,
        ):
            raise RuntimeError(
                "The first difficulty row is not exactly zero."
            )
        if not torch.allclose(
            terrain.difficulty_grid[-1],  # type: ignore
            expected_last,
        ):
            raise RuntimeError(
                "The final difficulty row is not exactly one."
            )

    if num_rows == 10 and num_cols == 40:
        # torch.bincount() 统计每个整数出现多少次。
        counts = torch.bincount(
            terrain.terrain_class_grid.reshape(-1),  # type: ignore
            minlength=5,
        )
        expected_counts = torch.full(
            (5,),
            80,
            dtype=counts.dtype,
            device=counts.device,
        )

        if not torch.equal(counts, expected_counts):
            raise RuntimeError(
                f"Expected 80 tiles per class, got {counts.tolist()}."
            )

        print(
            "[PASS] Full 10x40 grid contains exactly "
            "80 tiles for each terrain class."
        )

    print(
        "[PASS] waypoints_grid:",
        tuple(terrain.waypoints_grid.shape),  # type: ignore
    )
    print(
        "[PASS] terrain_class_grid:",
        tuple(terrain.terrain_class_grid.shape),  # type: ignore
    )
    print(
        "[PASS] difficulty_grid:",
        tuple(terrain.difficulty_grid.shape),  # type: ignore
    )


def main() -> None:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=0.01,
            device=args_cli.device,
        )
    )

    if args_cli.terrain_mode == "training":
        sim.set_camera_view(
            eye=[-105.0, -95.0, 115.0],  # type: ignore
            target=[0.0, 0.0, 0.0],  # type: ignore
        )
    else:
        sim.set_camera_view(
            eye=[-12.0, -12.0, 8.0],  # type: ignore
            target=[0.0, 0.0, 0.0],  # type: ignore
        )

    terrain, num_rows, num_cols = design_scene()
    sim.reset()

    waypoint_visualizer = None

    if args_cli.terrain_mode == "training":
        validate_parkour_metadata(
            terrain,
            num_rows,
            num_cols,
        )

        print("[INFO] terrain classes:")
        for class_index, class_name in enumerate(
            terrain.terrain_class_names  # type: ignore
        ):
            print(f"  {class_index}: {class_name}")

        if args_cli.show_waypoints:
            marker_cfg = SPHERE_MARKER_CFG.replace(  # type: ignore
                prim_path="/Visuals/ExtremeParkourWaypoints"
            )
            marker_cfg.markers["sphere"].radius = 0.06

            waypoint_visualizer = VisualizationMarkers(
                marker_cfg
            )
            waypoint_visualizer.visualize(
                translations=terrain.waypoints_grid.reshape(-1, 3)  # type: ignore
            )

            print(
                "[INFO] Showing "
                f"{terrain.waypoints_grid.numel() // 3} waypoints."  # type: ignore
            )

        print("[INFO] Extreme Parkour training terrain is ready.")
    else:
        print("[INFO] Continuous mixed parkour terrain is ready.")

    print(
        "[INFO] terrain_origins shape:",
        tuple(terrain.terrain_origins.shape),  # type: ignore
    )
    print(
        "[INFO] env_origins shape:",
        tuple(terrain.env_origins.shape),
    )

    step_count = 0
    while (
        simulation_app.is_running()
        and (
            args_cli.max_steps < 0
            or step_count < args_cli.max_steps
        )
    ):
        sim.step()
        step_count += 1

    # Keep a Python reference alive for the whole simulation loop.
    _ = waypoint_visualizer


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
