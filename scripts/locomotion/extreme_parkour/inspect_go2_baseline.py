#!/usr/bin/env python3
"""Validate the stage-one Go2 timing, actuator and runtime-name contract.

This development utility creates one Go2 articulation on a plane.  It does
not create the parkour terrain, a manager-based environment or an RL runner.

Example:

    python scripts/locomotion/extreme_parkour/inspect_go2_baseline.py --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect the Extreme Parkour Go2 baseline contract.")
parser.add_argument("--zero_policy_steps", type=int, default=50, help="Number of zero-action policy steps.")
parser.add_argument("--random_policy_steps", type=int, default=50, help="Number of random-action policy steps.")
parser.add_argument(
    "--random_action_amplitude",
    type=float,
    default=0.20,
    help="Uniform normalized-action amplitude used in the random-action phase.",
)
parser.add_argument("--seed", type=int, default=0, help="Torch seed for deterministic random actions.")
parser.add_argument("--keep_alive", action="store_true", help="Keep the GUI alive after validation.")
parser.add_argument(
    "--check_waypoint_command",
    action="store_true",
    help=(
        "Create a one-environment Manager-Based parkour task "
        "and validate waypoint switching."
    ),
)
parser.add_argument(
    "--check_proprio_contract",
    action="store_true",
    help=(
        "Create a small Manager-Based parkour task and validate "
        "the official 53-D proprio observation contract."
    ),
)
parser.add_argument(
    "--check_terrain_scan_contract",
    action="store_true",
    help=(
        "Create a small Manager-Based parkour task and validate "
        "the official 132-D terrain-scan contract."
    ),
)
parser.add_argument(
    "--check_priv_explicit_contract",
    action="store_true",
    help=(
        "Create a small Manager-Based parkour task and validate "
        "the official 9-D explicit privileged observation."
    ),
)
parser.add_argument(
    "--check_priv_latent_contract",
    action="store_true",
    help=(
        "Create a small Manager-Based parkour task and validate "
        "the official 29-D latent privileged observation."
    ),
)
parser.add_argument(
    "--check_proprio_history_contract",
    action="store_true",
    help=(
        "Create a small Manager-Based parkour task and validate "
        "the official 10 x 53 = 530 proprio-history contract."
    ),
)
parser.add_argument(
    "--check_navigation_rewards",
    action="store_true",
    help=(
        "Validate the Extreme Parkour waypoint velocity "
        "and yaw rewards with controlled states."
    ),
)
parser.add_argument(
    "--check_body_regularization_rewards",
    action="store_true",
    help=(
        "Validate Extreme Parkour vertical-velocity, "
        "angular-velocity and orientation penalties."
    ),
)
parser.add_argument(
    "--check_joint_regularization_rewards",
    action="store_true",
    help=(
        "Validate Extreme Parkour policy-step joint acceleration, "
        "action-rate, torque-change and torque penalties."
    ),
)
parser.add_argument(
    "--check_contact_pose_rewards",
    action="store_true",
    help=(
        "Validate Extreme Parkour collision, hip-position, "
        "joint-pose and foot-stumble penalties."
    ),
)
parser.add_argument(
    "--check_feet_edge_reward",
    action="store_true",
    help=(
        "Validate the Extreme Parkour foot-edge detector, "
        "terrain-level gate and one-step contact hysteresis."
    ),
)
parser.add_argument(
    "--check_positive_reward_clipping",
    action="store_true",
    help=(
        "Validate official only-positive total reward clipping "
        "while preserving raw per-term reward logs."
    ),
)

parser.add_argument(
    "--check_push_randomization",
    action="store_true",
    help=(
        "Validate the 8-second planar velocity push."
    ),
)

parser.add_argument(
    "--check_action_delay",
    action="store_true",
    help=(
        "Validate immediate and fixed 20-ms actions."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv

from blank_rl_lab.assets.robot.unitree import UNITREE_GO2_CFG  # noqa: E402
from blank_rl_lab.tasks.manager_based.locomotion.legged.velocity.go2_extreme_parkour_env_cfg import (  # noqa: E402
    EXTREME_PARKOUR_ACTION_SCALE,
    EXTREME_PARKOUR_DECIMATION,
    EXTREME_PARKOUR_GO2_CFG,
    EXTREME_PARKOUR_HEIGHT_SCAN_UPDATE_PERIOD,
    EXTREME_PARKOUR_NUM_ACTIONS,
    EXTREME_PARKOUR_POLICY_DT,
    EXTREME_PARKOUR_POLICY_FOOT_NAMES,
    EXTREME_PARKOUR_POLICY_JOINT_NAMES,
    EXTREME_PARKOUR_SIM_DT,
    ExtremeParkourManagerBasedRLEnv,
    Go2ExtremeParkourTeacherEnvCfg,
)

EXPECTED_FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

def _print_names(title: str, names: list[str]) -> None:
    print(f"\n{title} ({len(names)}):")
    for index, name in enumerate(names):
        print(f"  [{index:02d}] {name}")

def _assert_finite(robot: Articulation, phase: str) -> None:
    tensors = {
        "root_pos_w": robot.data.root_pos_w,
        "root_quat_w": robot.data.root_quat_w,
        "joint_pos": robot.data.joint_pos,
        "joint_vel": robot.data.joint_vel,
    }
    bad = [name for name, value in tensors.items() if not torch.isfinite(value).all()]
    if bad:
        raise RuntimeError(f"Non-finite robot state after {phase}: {bad}")

def _reset_to_default(robot: Articulation) -> None:
    root_state = robot.data.default_root_state.clone()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()

def _step_policy(robot: Articulation, sim: sim_utils.SimulationContext, actions: torch.Tensor) -> None:
    target = robot.data.default_joint_pos + EXTREME_PARKOUR_ACTION_SCALE * actions
    robot.set_joint_position_target(target)
    for _ in range(EXTREME_PARKOUR_DECIMATION):
        robot.write_data_to_sim()
        sim.step()
        robot.update(EXTREME_PARKOUR_SIM_DT)

def _write_robot_above_waypoint(
    env: ManagerBasedRLEnv,
    waypoint_positions_w: torch.Tensor,
    xy_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Place each robot base directly above a waypoint."""

    robot: Articulation = env.scene["robot"]

    root_state_w = robot.data.root_state_w.clone()

    root_state_w[:, 0] = waypoint_positions_w[:, 0] + xy_offset[0]

    root_state_w[:, 1] =  waypoint_positions_w[:, 1] + xy_offset[1]

    # waypoint z is the terrain surface height.
    # default_root_state z is the nominal base clearance.
    root_state_w[:, 2] = waypoint_positions_w[:, 2] + robot.data.default_root_state[:, 2]

    # Restore the configured upright reset orientation.
    root_state_w[:, 3:7] = robot.data.default_root_state[:, 3:7]

    # Remove linear and angular velocity after teleportation.
    root_state_w[:, 7:13] = 0.0

    robot.write_root_pose_to_sim(root_state_w[:, :7])
    robot.write_root_velocity_to_sim(root_state_w[:, 7:13])

def _step_at_waypoint(
    env: ManagerBasedRLEnv,
    waypoint_positions_w: torch.Tensor,
    zero_actions: torch.Tensor,
    xy_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Teleport above a waypoint and execute one policy step."""

    _write_robot_above_waypoint(env, waypoint_positions_w, xy_offset=xy_offset,)
    env.step(zero_actions)

def _validate_waypoint_command() -> None:
    """Validate waypoint hold, switching, completion and route sync."""

    env_cfg = Go2ExtremeParkourTeacherEnvCfg()

    # Keep this development test small.
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device

    terrain_generator_cfg = env_cfg.scene.terrain.terrain_generator
    if terrain_generator_cfg is None:
        raise RuntimeError(
            "Extreme Parkour requires a terrain generator."
        )

    # Two difficulty rows and one column per terrain family are
    # sufficient for command/curriculum validation.
    terrain_generator_cfg.num_rows = 2
    terrain_generator_cfg.num_cols = 5
    terrain_generator_cfg.curriculum = True

    env_cfg.scene.terrain.max_init_terrain_level = 0

    # Markers are useful in GUI mode but unnecessary headlessly.
    env_cfg.commands.waypoint.debug_vis = not args_cli.headless

    env = ManagerBasedRLEnv(cfg=env_cfg)

    try:
        env.reset()

        command_term = env.command_manager.get_term("waypoint")
        terrain = env.scene.terrain

        zero_actions = torch.zeros_like(env.action_manager.action)

        hold_steps = math.ceil(
            (command_term.cfg.reach_hold_time - 1.0e-6) / env.step_dt #type:ignore
        )

        if hold_steps != 5:
            raise RuntimeError(
                "Expected five policy steps for the "
                f"0.1 s hold, got {hold_steps}."
            )

        print("[INFO] policy step_dt:", env.step_dt,)
        print("[INFO] required hold steps:", hold_steps,)
        print(
            "[INFO] initial terrain level/type:",
            int(terrain.terrain_levels[0].item()), # type:ignore
            int(terrain.terrain_types[0].item()),  # type:ignore
        )

        initial_index = int(command_term.waypoint_index[0].item()) # type:ignore

        if initial_index != 0:
            raise RuntimeError(
                f"Expected initial waypoint index 0, "
                f"got {initial_index}."
            )

        first_waypoint_w = command_term.current_waypoint_w.clone() # type:ignore

        # Part 1:
        # Staying for only four steps must not advance.
        for step_index in range(hold_steps - 1):
            _step_at_waypoint(
                env,
                first_waypoint_w,
                zero_actions,
            )

            actual_index = int(command_term.waypoint_index[0].item()) # type:ignore
            if actual_index != 0:
                raise RuntimeError(
                    "Waypoint advanced too early at hold step "
                    f"{step_index + 1}: index={actual_index}."
                )

        print("[PASS] Four policy steps do not advance the waypoint.")

        # Part 2:
        # Leaving the target radius must clear the timer.
        outside_offset = (
            command_term.cfg.reach_threshold + 0.5, # type:ignore
            0.0,
        )
        _step_at_waypoint(
            env,
            first_waypoint_w,
            zero_actions,
            xy_offset=outside_offset,
        )

        reach_time = float(command_term.reach_time[0].item()) # type:ignore

        if abs(reach_time) > 1.0e-6:
            raise RuntimeError(
                "Leaving the target did not reset reach_time: "
                f"{reach_time}."
            )

        if int(
            command_term.waypoint_index[0].item() # type:ignore
        ) != 0:
            raise RuntimeError(
                "Waypoint changed after leaving the target."
            )

        print("[PASS] Leaving the target clears the continuous-hold timer.")

        # Part 3:
        # Five new continuous steps must advance exactly once.
        for step_index in range(hold_steps):
            _step_at_waypoint(
                env,
                first_waypoint_w,
                zero_actions,
            )

            if step_index < hold_steps - 1:
                actual_index = int(
                    command_term.waypoint_index[0].item() # type:ignore
                )
                if actual_index != 0:
                    raise RuntimeError(
                        "Waypoint advanced before completing "
                        "the new continuous hold."
                    )

        actual_index = int(command_term.waypoint_index[0].item()) # type:ignore
        if actual_index != 1:
            raise RuntimeError(
                "Expected waypoint index 1 after the full "
                f"hold, got {actual_index}."
            )

        print("[PASS] Five continuous policy steps advance the waypoint exactly once.")

        # Part 4:
        # Remaining at the old waypoint must not advance again.
        _step_at_waypoint(
            env,
            first_waypoint_w,
            zero_actions,
        )

        if int(command_term.waypoint_index[0].item()) != 1: # type:ignore
            raise RuntimeError(
                "Waypoint advanced more than once while the "
                "robot remained at the old waypoint."
            )

        print(
            "[PASS] Old waypoint cannot trigger a second "
            "increment."
        )

        # Part 5:
        # Complete waypoints 1 through 7.
        while int(command_term.waypoint_index[0].item()) < command_term.cfg.num_waypoints: # type:ignore

            index_before = int(command_term.waypoint_index[0].item()) # type:ignore
            target_waypoint_w = (command_term.current_waypoint_w.clone()) # type:ignore

            for _ in range(hold_steps):
                _step_at_waypoint(
                    env,
                    target_waypoint_w,
                    zero_actions,
                )

            index_after = int(command_term.waypoint_index[0].item()) # type:ignore

            if index_after != index_before + 1:
                raise RuntimeError(
                    "Waypoint did not advance exactly once: "
                    f"{index_before} -> {index_after}."
                )

            print(
                "[PASS] waypoint",
                index_before,
                "->",
                index_after,
            )

        if not bool(command_term.route_complete[0].item()): # type:ignore
            raise RuntimeError("route_complete is False after waypoint 8.")

        print("[PASS] All eight waypoints completed.")

        # route_complete becomes true during command update.
        # TerminationManager observes it on the following step.
        previous_level = int(terrain.terrain_levels[0].item()) # type:ignore

        env.step(zero_actions)

        new_level = int(terrain.terrain_levels[0].item()) # type:ignore
        new_type = int(
            terrain.terrain_types[0].item() # type:ignore
        )

        if int(command_term.waypoint_index[0].item()) != 0: # type:ignore
            raise RuntimeError("Waypoint index was not reset to zero.")

        if bool(command_term.route_complete[0].item()): # type:ignore
            raise RuntimeError("route_complete was not cleared on reset.")

        if int(command_term.terrain_level[0].item()) != new_level: # type:ignore
            raise RuntimeError("Command terrain_level is stale after curriculum update.")

        if int(command_term.terrain_type[0].item()) != new_type: # type:ignore
            raise RuntimeError("Command terrain_type is stale after reset.")

        expected_route = terrain.waypoints_grid[new_level, new_type,] # type:ignore

        if not torch.allclose(command_term.route_waypoints_w[0], expected_route,): # type:ignore
            raise RuntimeError("Command still contains waypoint data from the previous terrain tile.")

        print(
            "[PASS] route reset:",
            f"level {previous_level} -> {new_level},",
            f"type={new_type}",
        )
        print(
            "[PASS] New terrain route was loaded without "
            "stale waypoint data."
        )

        if args_cli.keep_alive:
            print(
                "[INFO] GUI kept alive. Green=current, "
                "blue=next. Close Isaac Sim to stop."
            )

            while simulation_app.is_running():
                with torch.inference_mode():
                    env.step(zero_actions)

    finally:
        env.close()

def _validate_proprio_contract() -> None:
    """Validate dimensions, slices, ordering and values of 53-D proprio."""

    env_cfg = Go2ExtremeParkourTeacherEnvCfg()

    # Five environments cover the five terrain columns while keeping this
    # inspection task lightweight.
    env_cfg.scene.num_envs = 5
    env_cfg.sim.device = args_cli.device

    terrain_generator_cfg = env_cfg.scene.terrain.terrain_generator
    if terrain_generator_cfg is None:
        raise RuntimeError("Extreme Parkour requires a terrain generator.")

    terrain_generator_cfg.num_rows = 2
    terrain_generator_cfg.num_cols = 5
    terrain_generator_cfg.curriculum = True
    env_cfg.scene.terrain.max_init_terrain_level = 0

    env = ManagerBasedRLEnv(cfg=env_cfg)

    try:
        observations, _ = env.reset()

        expected_term_names = [
            "base_ang_vel",
            "base_roll_pitch",
            "navigation",
            "joint_pos",
            "joint_vel",
            "last_action",
            "foot_contacts",
        ]
        expected_term_dims = [
            (3,),
            (2,),
            (8,),
            (12,),
            (12,),
            (12,),
            (4,),
        ]

        actual_term_names = env.observation_manager.active_terms.get("proprio")
        if actual_term_names != expected_term_names:
            raise RuntimeError(
                "Incorrect proprio term order:\n"
                f"expected={expected_term_names}\n"
                f"actual={actual_term_names}"
            )

        actual_term_dims = env.observation_manager.group_obs_term_dim["proprio"]
        if actual_term_dims != expected_term_dims:
            raise RuntimeError(
                "Incorrect proprio term dimensions:\n"
                f"expected={expected_term_dims}\n"
                f"actual={actual_term_dims}"
            )

        proprio = observations["proprio"]
        if proprio.shape != (env.num_envs, 53): # type:ignore
            raise RuntimeError(
                f"Expected proprio shape ({env.num_envs}, 53), "
                f"got {tuple(proprio.shape)}." # type:ignore
            )
        if not torch.isfinite(proprio).all(): # type:ignore
            raise RuntimeError("Initial proprio contains NaN or Inf.")

        print("[PASS] Proprio group shape is [num_envs, 53].")
        print("[PASS] Observation term order and dimensions are correct.")

        # Validate action-space joint ordering.
        action_term = env.action_manager.get_term("joint_pos")
        actual_joint_names = tuple(action_term._joint_names) # type:ignore
        if actual_joint_names != EXTREME_PARKOUR_POLICY_JOINT_NAMES:
            raise RuntimeError(
                "Action joint order does not match policy order:\n"
                f"expected={EXTREME_PARKOUR_POLICY_JOINT_NAMES}\n"
                f"actual={actual_joint_names}"
            )

        contact_sensor = env.scene.sensors["contact_forces"]
        _, resolved_foot_names = contact_sensor.find_bodies( # type:ignore
            list(EXTREME_PARKOUR_POLICY_FOOT_NAMES),
            preserve_order=True,
        )
        if tuple(resolved_foot_names) != EXTREME_PARKOUR_POLICY_FOOT_NAMES:
            raise RuntimeError(
                "Foot contact order does not match policy order:\n"
                f"expected={EXTREME_PARKOUR_POLICY_FOOT_NAMES}\n"
                f"actual={tuple(resolved_foot_names)}"
            )

        print("[PASS] Action joint order is FR, FL, RR, RL.")
        print("[PASS] Foot contact order is FR, FL, RR, RL.")

        # Give every action channel a different value so an incorrect
        # permutation cannot accidentally pass.
        test_action_single = torch.linspace(
            -0.55,
            0.55,
            EXTREME_PARKOUR_NUM_ACTIONS,
            device=env.device,
        )
        test_actions = test_action_single.unsqueeze(0).repeat(env.num_envs, 1)

        observations, _, _, _, _ = env.step(test_actions)
        proprio = observations["proprio"]

        if proprio.shape != (env.num_envs, 53): # type:ignore
            raise RuntimeError(
                "Proprio shape changed after stepping: "
                f"{tuple(proprio.shape)}." # type:ignore
            )
        if not torch.isfinite(proprio).all(): # type:ignore
            raise RuntimeError("Stepped proprio contains NaN or Inf.")

        robot: Articulation = env.scene["robot"]
        joint_ids = action_term._joint_ids # type:ignore
        roll, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)

        expected_values = {
            "base_ang_vel": robot.data.root_ang_vel_b * 0.25,
            "base_roll_pitch": torch.stack((roll, pitch), dim=-1),
            "navigation": env.command_manager.get_command("waypoint"),
            "joint_pos": (
                robot.data.joint_pos[:, joint_ids]
                - robot.data.default_joint_pos[:, joint_ids]
            ),
            "joint_vel": (
                robot.data.joint_vel[:, joint_ids]
                - robot.data.default_joint_vel[:, joint_ids]
            )
            * 0.05,
            "last_action": test_actions,
        }
        proprio_slices = {
            "base_ang_vel": slice(0, 3),
            "base_roll_pitch": slice(3, 5),
            "navigation": slice(5, 13),
            "joint_pos": slice(13, 25),
            "joint_vel": slice(25, 37),
            "last_action": slice(37, 49),
            "foot_contacts": slice(49, 53),
        }

        for name, expected in expected_values.items():
            actual = proprio[:, proprio_slices[name]] # type:ignore
            torch.testing.assert_close(
                actual,
                expected,
                rtol=1.0e-5,
                atol=1.0e-6,
                msg=(
                    f"Incorrect values in proprio term '{name}' "
                    f"at slice {proprio_slices[name]}."
                ),
            )

        print("[PASS] All non-contact proprio slices match source data.")

        # Validate target = default position + 0.25 * policy action.
        expected_processed_actions = (
            robot.data.default_joint_pos[:, joint_ids]
            + EXTREME_PARKOUR_ACTION_SCALE * test_actions
        )
        torch.testing.assert_close(
            action_term.processed_actions,
            expected_processed_actions,
            rtol=1.0e-6,
            atol=1.0e-6,
            msg="Processed joint targets do not match policy order.",
        )
        print("[PASS] Action mapping is default_joint_pos + 0.25 * policy_action.")

        # Official contact encoding is -0.5 for no contact and +0.5
        # for contact.
        foot_contacts = proprio[:, proprio_slices["foot_contacts"]] # type:ignore
        valid_contact_values = torch.logical_or(
            foot_contacts == -0.5,
            foot_contacts == 0.5,
        )
        if not torch.all(valid_contact_values):
            raise RuntimeError(
                "Foot contacts must contain only -0.5 or +0.5, "
                f"got {torch.unique(foot_contacts).tolist()}."
            )
        print("[PASS] Foot contacts are encoded as -0.5 or +0.5.")

        # Validate navigation placeholders, yaw ranges, forward speed and
        # the local terrain-class mapping.
        navigation = proprio[:, proprio_slices["navigation"]] # type:ignore
        zero_channels = navigation[:, [0, 3, 4]]
        if not torch.allclose(
            zero_channels,
            torch.zeros_like(zero_channels),
            atol=1.0e-6,
        ):
            raise RuntimeError(
                "Navigation placeholder channels 0, 3 and 4 must be zero."
            )
        if torch.any(torch.abs(navigation[:, 1:3]) > torch.pi + 1.0e-6):
            raise RuntimeError(
                "Current/next waypoint yaw errors are outside [-pi, pi]."
            )

        command_term = env.command_manager.get_term("waypoint")
        speed_min, speed_max = command_term.cfg.forward_speed_range # type:ignore
        if torch.any(navigation[:, 5] < speed_min) or torch.any(
            navigation[:, 5] > speed_max
        ):
            raise RuntimeError(
                "Navigation forward speed is outside "
                f"[{speed_min}, {speed_max}]."
            )

        is_flat = command_term.terrain_class == command_term.cfg.parkour_flat_terrain_class # type:ignore

        torch.testing.assert_close(
            navigation[:, 6],
            (~is_flat).float(),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            navigation[:, 7],
            is_flat.float(),
            rtol=0.0,
            atol=0.0,
        )

        print("[PASS] Navigation placeholders and yaw ranges are correct.")
        print("[PASS] Local terrain class 2 is encoded as flat.")
        print("[PASS] Official 53-D proprio contract validated.")

        print("\nProprio slice contract:")
        for name, term_slice in proprio_slices.items():
            print(
                f"  {name:18s}: "
                f"[{term_slice.start:02d}:{term_slice.stop:02d}]"
            )

    finally:
        env.close()

def _validate_terrain_scan_contract() -> None:
    """Validate the official 12 x 11 terrain-scan contract."""

    env_cfg = Go2ExtremeParkourTeacherEnvCfg()

    # One environment for each terrain family.
    env_cfg.scene.num_envs = 5
    env_cfg.sim.device = args_cli.device

    terrain_generator_cfg = env_cfg.scene.terrain.terrain_generator
    if terrain_generator_cfg is None:
        raise RuntimeError("Extreme Parkour requires a terrain generator.")

    terrain_generator_cfg.num_rows = 2
    terrain_generator_cfg.num_cols = 5
    terrain_generator_cfg.curriculum = True
    env_cfg.scene.terrain.max_init_terrain_level = 0

    env = ManagerBasedRLEnv(cfg=env_cfg)

    try:
        observations, _ = env.reset()

        # ------------------------------------------------------------------
        # 1. ObservationManager group contract
        # ------------------------------------------------------------------
        actual_term_names = env.observation_manager.active_terms.get("terrain_scan")

        expected_term_names = ["height_scan"]

        if actual_term_names != expected_term_names:
            raise RuntimeError(
                "Incorrect terrain_scan term order:\n"
                f"expected={expected_term_names}\n"
                f"actual={actual_term_names}"
            )

        actual_term_dims = env.observation_manager.group_obs_term_dim["terrain_scan"]

        expected_term_dims = [(132,)]

        if actual_term_dims != expected_term_dims:
            raise RuntimeError(
                "Incorrect terrain_scan term dimensions:\n"
                f"expected={expected_term_dims}\n"
                f"actual={actual_term_dims}"
            )

        terrain_scan = observations["terrain_scan"]

        if terrain_scan.shape != (env.num_envs, 132): #type:ignore
            raise RuntimeError(
                f"Expected terrain_scan shape ({env.num_envs}, 132), "
                f"got {tuple(terrain_scan.shape)}." #type:ignore
            )

        print("[PASS] terrain_scan group shape is [num_envs, 132].")
        print("[PASS] terrain_scan contains exactly one 132-D term.")

        # ------------------------------------------------------------------
        # 2. RayCaster local-coordinate contract
        # ------------------------------------------------------------------
        height_scanner = env.scene.sensors["height_scanner"]

        if height_scanner.num_rays != 132: # type:ignore
            raise RuntimeError(f"Expected 132 height rays, got {height_scanner.num_rays}.") # type:ignore

        expected_x = torch.linspace(
            -0.45,
            1.20,
            12,
            device=env.device,
        )
        expected_y = torch.linspace(
            -0.75,
            0.75,
            11,
            device=env.device,
        )

        # indexing="ij" means that y changes inside each x row:
        # (-0.45, -0.75), (-0.45, -0.60), ..., (-0.30, -0.75)
        grid_x, grid_y = torch.meshgrid(
            expected_x,
            expected_y,
            indexing="ij",
        )

        expected_ray_starts = torch.zeros(
            132,
            3,
            device=env.device,
        )
        expected_ray_starts[:, 0] = grid_x.flatten()
        expected_ray_starts[:, 1] = grid_y.flatten()
        expected_ray_starts[:, 2] = 20.0

        actual_ray_starts = height_scanner.ray_starts[0] # type:ignore

        torch.testing.assert_close(
            actual_ray_starts,
            expected_ray_starts,
            rtol=0.0,
            atol=1.0e-6,
            msg="RayCaster local starting points do not match the official grid.",
        )

        expected_ray_directions = torch.zeros_like(actual_ray_starts)
        expected_ray_directions[:, 2] = -1.0

        torch.testing.assert_close(
            height_scanner.ray_directions[0], # type:ignore
            expected_ray_directions,
            rtol=0.0,
            atol=1.0e-6,
            msg="Height rays must point vertically downward.",
        )

        print("[PASS] Ray grid is 12 x 11 in official flattening order.")
        print("[PASS] Ray x range is [-0.45, 1.20] m.")
        print("[PASS] Ray y range is [-0.75, 0.75] m.")
        print("[PASS] All 132 rays point vertically downward.")

        # ------------------------------------------------------------------
        # 3. Validate the observation formula and clipping
        # ------------------------------------------------------------------
        def validate_scan_values(
            current_observations: dict,
            phase: str,
        ) -> torch.Tensor:
            current_scan = current_observations["terrain_scan"]

            if current_scan.shape != (env.num_envs, 132):
                raise RuntimeError(
                    f"{phase}: terrain_scan has incorrect shape "
                    f"{tuple(current_scan.shape)}."
                )

            if not torch.isfinite(current_scan).all():
                raise RuntimeError(
                    f"{phase}: terrain_scan contains NaN or Inf."
                )

            if torch.any(current_scan < -1.0) or torch.any(
                current_scan > 1.0
            ):
                raise RuntimeError(
                    f"{phase}: terrain_scan is outside [-1, 1]."
                )

            sensor_data = height_scanner.data

            expected_scan = (
                sensor_data.pos_w[:, 2].unsqueeze(1)
                - sensor_data.ray_hits_w[..., 2]
                - 0.3
            )
            expected_scan = expected_scan.clamp(
                min=-1.0,
                max=1.0,
            )

            torch.testing.assert_close(
                current_scan,
                expected_scan,
                rtol=1.0e-5,
                atol=1.0e-6,
                msg=(
                    f"{phase}: terrain_scan does not equal "
                    "base_z - hit_z - 0.3."
                ),
            )

            return current_scan

        initial_scan = validate_scan_values(
            observations,
            phase="initial observation",
        ).clone()

        print("[PASS] terrain_scan equals clip(base_z - hit_z - 0.3).")
        print("[PASS] terrain_scan contains no NaN or Inf.")
        print("[PASS] terrain_scan is clipped to [-1, 1].")

        # ------------------------------------------------------------------
        # 4. Validate the 10 Hz update period
        # ------------------------------------------------------------------
        actual_update_period = height_scanner.cfg.update_period

        if not math.isclose(
            actual_update_period,
            EXTREME_PARKOUR_HEIGHT_SCAN_UPDATE_PERIOD,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError(
                "Incorrect height-scanner update period: "
                f"expected={EXTREME_PARKOUR_HEIGHT_SCAN_UPDATE_PERIOD}, "
                f"actual={actual_update_period}."
            )

        update_steps = round(actual_update_period / env.step_dt)

        if update_steps != 5:
            raise RuntimeError(
                "Expected one scan update every five policy steps, "
                f"got {update_steps} steps."
            )

        zero_actions = torch.zeros_like(env.action_manager.action)

        # This is an internal sensor timestamp, used only by this development
        # contract test. Production observations do not access private fields.
        initial_update_timestamp = (
            height_scanner._timestamp_last_update.clone() # type:ignore
        )

        # The first four policy steps must keep the old scan.
        for step_index in range(update_steps - 1):
            observations, _, _, _, _ = env.step(zero_actions)

            current_scan = validate_scan_values(
                observations,
                phase=f"hold step {step_index + 1}",
            )

            current_timestamp = (
                height_scanner._timestamp_last_update # type:ignore
            )

            if not torch.allclose(
                current_timestamp,
                initial_update_timestamp,
                rtol=0.0,
                atol=1.0e-7,
            ):
                raise RuntimeError(
                    "Height scanner updated before the fifth policy step."
                )

            if not torch.equal(current_scan, initial_scan):
                raise RuntimeError(
                    "terrain_scan changed while the 10 Hz sensor "
                    "should have held its previous value."
                )

        print("[PASS] terrain_scan is held for the first four policy steps.")

        # The fifth policy step must refresh the RayCaster data.
        observations, _, _, _, _ = env.step(zero_actions)

        validate_scan_values(observations, phase="fifth policy step",)

        refreshed_timestamp = height_scanner._timestamp_last_update # type:ignore

        if not torch.all(refreshed_timestamp > initial_update_timestamp):
            raise RuntimeError(
                "Height scanner did not refresh on the fifth policy step."
            )

        timestamp_delta = (
            refreshed_timestamp - initial_update_timestamp
        )

        torch.testing.assert_close(
            timestamp_delta,
            torch.full_like(
                timestamp_delta,
                EXTREME_PARKOUR_HEIGHT_SCAN_UPDATE_PERIOD,
            ),
            rtol=0.0,
            atol=1.0e-6,
            msg="Height-scanner timestamp advanced by the wrong duration.",
        )

        print("[PASS] Height scanner refreshes on policy step five.")
        print("[PASS] Height scanner update frequency is 10 Hz.")
        print("[PASS] Official 132-D terrain-scan contract validated.")

    finally:
        env.close()

def _validate_priv_explicit_contract() -> None:
    """Validate the official 9-D explicit privileged observation."""

    env_cfg = Go2ExtremeParkourTeacherEnvCfg()

    env_cfg.scene.num_envs = 5
    env_cfg.sim.device = args_cli.device

    terrain_generator_cfg = env_cfg.scene.terrain.terrain_generator
    if terrain_generator_cfg is None:
        raise RuntimeError("Extreme Parkour requires a terrain generator.")

    terrain_generator_cfg.num_rows = 2
    terrain_generator_cfg.num_cols = 5
    terrain_generator_cfg.curriculum = True
    env_cfg.scene.terrain.max_init_terrain_level = 0

    env = ManagerBasedRLEnv(cfg=env_cfg)

    try:
        observations, _ = env.reset()

        # ------------------------------------------------------------------
        # 1. ObservationManager metadata contract
        # ------------------------------------------------------------------
        expected_term_names = [
            "base_lin_vel",
            "base_lin_vel_placeholder_1",
            "base_lin_vel_placeholder_2",
        ]
        actual_term_names = env.observation_manager.active_terms.get(
            "priv_explicit"
        )

        if actual_term_names != expected_term_names:
            raise RuntimeError(
                "Incorrect priv_explicit term order:\n"
                f"expected={expected_term_names}\n"
                f"actual={actual_term_names}"
            )

        expected_term_dims = [
            (3,),
            (3,),
            (3,),
        ]
        actual_term_dims = env.observation_manager.group_obs_term_dim["priv_explicit"]


        if actual_term_dims != expected_term_dims:
            raise RuntimeError(
                "Incorrect priv_explicit term dimensions:\n"
                f"expected={expected_term_dims}\n"
                f"actual={actual_term_dims}"
            )

        priv_explicit = observations["priv_explicit"]

        if priv_explicit.shape != (env.num_envs, 9): # type:ignore
            raise RuntimeError(
                f"Expected priv_explicit shape ({env.num_envs}, 9), "
                f"got {tuple(priv_explicit.shape)}." # type:ignore
            )

        if not torch.isfinite(priv_explicit).all(): # type:ignore
            raise RuntimeError(
                "Initial priv_explicit contains NaN or Inf."
            )

        print("[PASS] priv_explicit group shape is [num_envs, 9].")
        print("[PASS] priv_explicit term order is 3 + 3 + 3.")

        # ------------------------------------------------------------------
        # 2. Validate values and slices
        # ------------------------------------------------------------------
        robot: Articulation = env.scene["robot"]

        priv_explicit_slices = {
            "base_lin_vel": slice(0, 3),
            "placeholder_1": slice(3, 6),
            "placeholder_2": slice(6, 9),
        }

        def validate_priv_explicit_values(
            current_observations: dict,
            phase: str,
        ) -> torch.Tensor:
            current_priv = current_observations["priv_explicit"]

            if current_priv.shape != (env.num_envs, 9):
                raise RuntimeError(
                    f"{phase}: incorrect priv_explicit shape "
                    f"{tuple(current_priv.shape)}."
                )

            if not torch.isfinite(current_priv).all():
                raise RuntimeError(
                    f"{phase}: priv_explicit contains NaN or Inf."
                )

            # ObservationManager applies clip before scale:
            # clip(root_lin_vel_b, -50, 50) * 2.
            expected_base_lin_vel = (
                robot.data.root_lin_vel_b
                .clamp(min=-50.0, max=50.0)
                * 2.0
            )

            torch.testing.assert_close(
                current_priv[:, priv_explicit_slices["base_lin_vel"],],
                expected_base_lin_vel,
                rtol=1.0e-5,
                atol=1.0e-6,
                msg=(
                    f"{phase}: priv_explicit[0:3] does not equal "
                    "2 * root_lin_vel_b."
                ),
            )

            placeholder_1 = current_priv[:, priv_explicit_slices["placeholder_1"],]
            placeholder_2 = current_priv[:, priv_explicit_slices["placeholder_2"],]

            if not torch.equal(
                placeholder_1,
                torch.zeros_like(placeholder_1),
            ):
                raise RuntimeError(
                    f"{phase}: placeholder_1 is not exactly zero."
                )

            if not torch.equal(
                placeholder_2,
                torch.zeros_like(placeholder_2),
            ):
                raise RuntimeError(
                    f"{phase}: placeholder_2 is not exactly zero."
                )

            return current_priv

        validate_priv_explicit_values(
            observations,
            phase="initial observation",
        )

        print("[PASS] priv_explicit[0:3] equals 2 * root_lin_vel_b.")
        print("[PASS] priv_explicit[3:9] contains exact zeros.")

        # ------------------------------------------------------------------
        # 3. Controlled coordinate-frame test
        # ------------------------------------------------------------------
        # Give the robot a +90 degree world yaw and a +X world velocity.
        # A world-frame implementation would report approximately [1, 0, 0].
        # A correct body-frame implementation reports approximately [0, -1, 0].
        root_pose_w = robot.data.root_com_pose_w.clone()

        zero_angle = torch.zeros(
            env.num_envs,
            device=env.device,
        )
        yaw_angle = torch.full(
            (env.num_envs,),
            torch.pi / 2.0,
            device=env.device,
        )

        root_pose_w[:, 3:7] = math_utils.quat_from_euler_xyz(
            zero_angle,
            zero_angle,
            yaw_angle,
        )

        root_velocity_w = torch.zeros(
            env.num_envs,
            6,
            device=env.device,
        )
        root_velocity_w[:, 0] = 1.0

        robot.write_root_pose_to_sim(root_pose_w)
        robot.write_root_velocity_to_sim(root_velocity_w)

        # No physics step is required: the write methods update both PhysX
        # and the articulation's internal root-state buffers.
        controlled_observations = (
            env.observation_manager.compute()
        )

        controlled_priv = validate_priv_explicit_values(
            controlled_observations,
            phase="controlled body-frame test",
        )

        # Independently reconstruct body-frame velocity from world-frame
        # velocity and the current root quaternion.
        expected_body_velocity = math_utils.quat_apply_inverse(
            robot.data.root_quat_w,
            robot.data.root_lin_vel_w,
        )

        torch.testing.assert_close(
            robot.data.root_lin_vel_b,
            expected_body_velocity,
            rtol=1.0e-5,
            atol=1.0e-6,
            msg=(
                "root_lin_vel_b does not match inverse quaternion "
                "rotation of root_lin_vel_w."
            ),
        )

        expected_controlled_observation = (
            expected_body_velocity
            .clamp(min=-50.0, max=50.0)
            * 2.0
        )

        torch.testing.assert_close(
            controlled_priv[:, 0:3],
            expected_controlled_observation,
            rtol=1.0e-5,
            atol=1.0e-6,
            msg=(
                "Controlled priv_explicit velocity is not expressed "
                "in the robot base frame."
            ),
        )

        # Ensure the fixture actually distinguishes world and body frames.
        if torch.allclose(
            robot.data.root_lin_vel_w,
            robot.data.root_lin_vel_b,
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise RuntimeError(
                "Controlled test did not distinguish world-frame and "
                "body-frame velocity."
            )

        print("[PASS] Controlled yaw test distinguishes world/body frames.")
        print("[PASS] Explicit velocity is expressed in the base frame.")
        print("[PASS] Velocity scale is exactly 2.0.")
        print("[PASS] Official 9-D priv_explicit contract validated.")

        print("\npriv_explicit slice contract:")
        for name, term_slice in priv_explicit_slices.items():
            print(
                f"  {name:18s}: "
                f"[{term_slice.start:02d}:{term_slice.stop:02d}]"
            )

    finally:
        env.close()

def _validate_priv_latent_contract() -> None:
    """Validate the official 29-D latent privileged observation."""

    env_cfg = Go2ExtremeParkourTeacherEnvCfg()

    env_cfg.scene.num_envs = 5
    env_cfg.sim.device = args_cli.device

    terrain_generator_cfg = env_cfg.scene.terrain.terrain_generator
    if terrain_generator_cfg is None:
        raise RuntimeError("Extreme Parkour requires a terrain generator.")

    terrain_generator_cfg.num_rows = 2
    terrain_generator_cfg.num_cols = 5
    terrain_generator_cfg.curriculum = True
    env_cfg.scene.terrain.max_init_terrain_level = 0

    env = ManagerBasedRLEnv(cfg=env_cfg)

    try:
        observations, _ = env.reset()

        # ------------------------------------------------------------------
        # 1. ObservationManager metadata
        # ------------------------------------------------------------------
        expected_term_names = [
            "mass_com",
            "friction",
            "stiffness_offset",
            "damping_offset",
        ]
        actual_term_names = env.observation_manager.active_terms.get(
            "priv_latent"
        )

        if actual_term_names != expected_term_names:
            raise RuntimeError(
                "Incorrect priv_latent term order:\n"
                f"expected={expected_term_names}\n"
                f"actual={actual_term_names}"
            )

        expected_term_dims = [
            (4,),
            (1,),
            (12,),
            (12,),
        ]
        actual_term_dims = (
            env.observation_manager.group_obs_term_dim["priv_latent"]
        )

        if actual_term_dims != expected_term_dims:
            raise RuntimeError(
                "Incorrect priv_latent term dimensions:\n"
                f"expected={expected_term_dims}\n"
                f"actual={actual_term_dims}"
            )

        priv_latent = observations["priv_latent"]

        if priv_latent.shape != (env.num_envs, 29): # type:ignore
            raise RuntimeError(
                f"Expected priv_latent shape ({env.num_envs}, 29), "
                f"got {tuple(priv_latent.shape)}." # type:ignore
            )

        if not torch.isfinite(priv_latent).all(): # type:ignore
            raise RuntimeError(
                "Initial priv_latent contains NaN or Inf."
            )

        print("[PASS] priv_latent group shape is [num_envs, 29].")
        print("[PASS] priv_latent term dimensions are 4 + 1 + 12 + 12.")

        # ------------------------------------------------------------------
        # 2. Resolve base body and policy joint mappings
        # ------------------------------------------------------------------
        robot: Articulation = env.scene["robot"]

        base_body_ids, base_body_names = robot.find_bodies(
            ["base"],
            preserve_order=True,
        )

        if tuple(base_body_names) != ("base",):
            raise RuntimeError(
                f"Expected one base body, got {base_body_names}."
            )

        base_body_id = base_body_ids[0]

        policy_joint_ids, policy_joint_names = robot.find_joints(
            list(EXTREME_PARKOUR_POLICY_JOINT_NAMES),
            preserve_order=True,
        )

        if tuple(policy_joint_names) != (
            EXTREME_PARKOUR_POLICY_JOINT_NAMES
        ):
            raise RuntimeError(
                "Policy joint order does not match FR, FL, RR, RL:\n"
                f"expected={EXTREME_PARKOUR_POLICY_JOINT_NAMES}\n"
                f"actual={tuple(policy_joint_names)}"
            )

        policy_joint_ids_tensor = torch.tensor(
            policy_joint_ids,
            dtype=torch.long,
            device=env.device,
        )

        actuator = robot.actuators["legs"]

        if isinstance(actuator.joint_indices, slice):
            actuator_global_joint_ids = torch.arange(
                robot.num_joints,
                dtype=torch.long,
                device=env.device,
            )[actuator.joint_indices]
        else:
            actuator_global_joint_ids = actuator.joint_indices.to(
                device=env.device,
                dtype=torch.long,
            )

        matches = (
            policy_joint_ids_tensor.unsqueeze(1)
            == actuator_global_joint_ids.unsqueeze(0)
        )

        if not torch.all(matches.sum(dim=1) == 1):
            raise RuntimeError(
                "Not every policy joint belongs to actuator 'legs'."
            )

        actuator_local_ids = matches.to(
            dtype=torch.long
        ).argmax(dim=1)

        print("[PASS] Base body resolved uniquely.")
        print("[PASS] Actuator channels resolve in FR, FL, RR, RL order.")

        # ------------------------------------------------------------------
        # 3. Slice contract and source-value helper
        # ------------------------------------------------------------------
        priv_latent_slices = {
            "mass_com": slice(0, 4),
            "friction": slice(4, 5),
            "stiffness_offset": slice(5, 17),
            "damping_offset": slice(17, 29),
        }

        default_stiffness = (
            robot.data.default_joint_stiffness[
                :,
                policy_joint_ids_tensor,
            ]
        )
        default_damping = (
            robot.data.default_joint_damping[
                :,
                policy_joint_ids_tensor,
            ]
        )

        def current_friction() -> torch.Tensor:
            materials = (
                robot.root_physx_view
                .get_material_properties()
                .to(env.device)
            )

            static_friction = materials[:, :, 0].mean(
                dim=1
            )
            dynamic_friction = materials[:, :, 1].mean(
                dim=1
            )

            return (
                0.5
                * (static_friction + dynamic_friction)
            ).unsqueeze(-1)

        def current_stiffness_offset() -> torch.Tensor:
            current_stiffness = actuator.stiffness[
                :,
                actuator_local_ids,
            ]

            return (
                current_stiffness
                / default_stiffness
                - 1.0
            )

        def current_damping_offset() -> torch.Tensor:
            current_damping = actuator.damping[
                :,
                actuator_local_ids,
            ]

            return (
                current_damping
                / default_damping
                - 1.0
            )

        def validate_priv_latent_values(
            current_observations: dict,
            expected_mass_com: torch.Tensor,
            phase: str,
        ) -> torch.Tensor:
            current_priv = current_observations["priv_latent"]

            if current_priv.shape != (env.num_envs, 29):
                raise RuntimeError(
                    f"{phase}: incorrect priv_latent shape "
                    f"{tuple(current_priv.shape)}."
                )

            if not torch.isfinite(current_priv).all():
                raise RuntimeError(
                    f"{phase}: priv_latent contains NaN or Inf."
                )

            expected_values = {
                "mass_com": expected_mass_com,
                "friction": current_friction(),
                "stiffness_offset": (
                    current_stiffness_offset()
                ),
                "damping_offset": (
                    current_damping_offset()
                ),
            }

            for name, expected in expected_values.items():
                actual = current_priv[
                    :,
                    priv_latent_slices[name],
                ]

                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                    msg=(
                        f"{phase}: incorrect priv_latent "
                        f"term '{name}'."
                    ),
                )

            return current_priv

        # ------------------------------------------------------------------
        # 4. Validate startup domain-randomization values
        # ------------------------------------------------------------------
        initial_priv = validate_priv_latent_values(
            observations,
            expected_mass_com=observations[ # type:ignore
                "priv_latent"
            ][:, 0:4].clone(),
            phase="startup randomization",
        )

        added_mass = initial_priv[:, 0]
        com_offset = initial_priv[:, 1:4]
        friction = initial_priv[:, 4]
        stiffness_offset = initial_priv[:, 5:17]
        damping_offset = initial_priv[:, 17:29]

        eps = 1.0e-6

        # 1. Added base mass must lie in [0, 3] kg.
        if torch.any(added_mass < -eps) or torch.any(
            added_mass > 3.0 + eps
        ):
            raise RuntimeError(
                "Startup added mass is outside [0, 3] kg."
            )

        current_base_mass = (
            robot.root_physx_view
            .get_masses()
            .to(env.device)[:, base_body_id]
        )

        default_base_mass = (
            robot.data.default_mass[
                :,
                base_body_id,
            ].to(env.device)
        )

        torch.testing.assert_close(
            current_base_mass - default_base_mass,
            added_mass,
            rtol=1.0e-5,
            atol=1.0e-6,
            msg=(
                "priv_latent added mass does not match "
                "the PhysX base mass."
            ),
        )

        # 2. CoM offset must lie in [-0.2, 0.2] m.
        if torch.any(com_offset < -0.2 - eps) or torch.any(
            com_offset > 0.2 + eps
        ):
            raise RuntimeError(
                "Startup CoM offset is outside [-0.2, 0.2] m."
            )

        current_base_com = (
            robot.root_physx_view
            .get_coms()
            .to(env.device)[
                :,
                base_body_id,
                :3,
            ]
        )

        # All cloned Go2 robots have the same nominal CoM.
        recovered_default_com = (
            current_base_com - com_offset
        )

        torch.testing.assert_close(
            recovered_default_com,
            recovered_default_com[0:1].expand_as(
                recovered_default_com
            ),
            rtol=1.0e-5,
            atol=1.0e-6,
            msg=(
                "CoM observation does not recover one "
                "consistent nominal Go2 CoM."
            ),
        )

        # 3. Every collision shape in one environment must use
        # exactly the same friction coefficient.
        materials = (
            robot.root_physx_view
            .get_material_properties()
            .to(env.device)
        )

        static_friction = materials[:, :, 0]
        dynamic_friction = materials[:, :, 1]

        torch.testing.assert_close(
            static_friction,
            static_friction[:, 0:1].expand_as(
                static_friction
            ),
            rtol=0.0,
            atol=1.0e-6,
            msg=(
                "One robot contains multiple static "
                "friction coefficients."
            ),
        )

        torch.testing.assert_close(
            dynamic_friction,
            static_friction,
            rtol=0.0,
            atol=1.0e-6,
            msg=(
                "Static and dynamic friction are not equal."
            ),
        )

        if torch.any(friction < 0.6 - eps) or torch.any(
            friction > 2.0 + eps
        ):
            raise RuntimeError(
                "Startup friction is outside [0.6, 2.0]."
            )

        print("[PASS] Added base mass is in [0, 3] kg.")
        print("[PASS] CoM offset is in [-0.2, 0.2] m.")
        print("[PASS] One friction coefficient is used per robot.")
        print("[PASS] Friction is in [0.6, 2.0].")

        # 4. Actuator-gain offsets must lie in [-0.2, 0.2].
        if torch.any(
            stiffness_offset < -0.2 - eps
        ) or torch.any(
            stiffness_offset > 0.2 + eps
        ):
            raise RuntimeError(
                "Stiffness offsets are outside [-0.2, 0.2]."
            )

        if torch.any(
            damping_offset < -0.2 - eps
        ) or torch.any(
            damping_offset > 0.2 + eps
        ):
            raise RuntimeError(
                "Damping offsets are outside [-0.2, 0.2]."
            )

        current_stiffness = actuator.stiffness[
            :,
            actuator_local_ids,
        ]

        current_damping = actuator.damping[
            :,
            actuator_local_ids,
        ]

        torch.testing.assert_close(
            current_stiffness,
            default_stiffness
            * (stiffness_offset + 1.0),
            rtol=1.0e-5,
            atol=1.0e-6,
            msg=(
                "Stiffness privileged observation does "
                "not match actuator.stiffness."
            ),
        )

        torch.testing.assert_close(
            current_damping,
            default_damping
            * (damping_offset + 1.0),
            rtol=1.0e-5,
            atol=1.0e-6,
            msg=(
                "Damping privileged observation does "
                "not match actuator.damping."
            ),
        )

        print("[PASS] Kp multipliers are in [0.8, 1.2].")
        print("[PASS] Kd multipliers are in [0.8, 1.2].")
        print("[PASS] Gain observations match actuator tensors.")

        # ------------------------------------------------------------------
        # 4. Controlled PhysX mass, CoM and friction test
        # ------------------------------------------------------------------
        env_ids_cpu = torch.arange(
            env.num_envs,
            dtype=torch.long,
            device="cpu",
        )

        masses = (
            robot.root_physx_view
            .get_masses()
            .clone()
        )

        nominal_base_mass = default_base_mass.to(
            masses.device
        )

        mass_delta = torch.linspace(
            0.1,
            0.5,
            env.num_envs,
            device=masses.device,
        )

        masses[:, base_body_id] = (
            nominal_base_mass + mass_delta
        )
        robot.root_physx_view.set_masses(
            masses,
            env_ids_cpu,
        )

        coms = (
            robot.root_physx_view
            .get_coms()
            .clone()
        )
        nominal_base_com = recovered_default_com.to(coms.device)

        com_delta = torch.zeros(
            env.num_envs,
            3,
            device=coms.device,
        )
        com_delta[:, 0] = torch.linspace(
            -0.10,
            0.10,
            env.num_envs,
            device=coms.device,
        )
        com_delta[:, 1] = torch.linspace(
            0.08,
            -0.08,
            env.num_envs,
            device=coms.device,
        )
        com_delta[:, 2] = torch.linspace(
            -0.04,
            0.04,
            env.num_envs,
            device=coms.device,
        )

        coms[:, base_body_id, :3] = (
            nominal_base_com + com_delta
        )
        robot.root_physx_view.set_coms(
            coms,
            env_ids_cpu,
        )

        materials = (
            robot.root_physx_view
            .get_material_properties()
            .clone()
        )

        controlled_friction = torch.linspace(
            0.6,
            1.4,
            env.num_envs,
            device=materials.device,
        )

        materials[:, :, 0] = (
            controlled_friction.unsqueeze(1)
        )
        materials[:, :, 1] = (
            controlled_friction.unsqueeze(1)
        )

        robot.root_physx_view.set_material_properties(
            materials,
            env_ids_cpu,
        )

        # ------------------------------------------------------------------
        # 5. Controlled explicit actuator-gain test
        # ------------------------------------------------------------------
        stiffness_multiplier = torch.linspace(
            0.81,
            1.19,
            12,
            device=env.device,
        ).unsqueeze(0).repeat(env.num_envs, 1)

        # Reverse the sequence so stiffness/damping ordering mistakes
        # cannot accidentally produce the same result.
        damping_multiplier = torch.linspace(
            1.18,
            0.82,
            12,
            device=env.device,
        ).unsqueeze(0).repeat(env.num_envs, 1)

        actuator.stiffness[
            :,
            actuator_local_ids,
        ] = (
            default_stiffness
            * stiffness_multiplier
        )

        actuator.damping[
            :,
            actuator_local_ids,
        ] = (
            default_damping
            * damping_multiplier
        )

        controlled_observations = (
            env.observation_manager.compute()
        )

        expected_mass_com = torch.cat(
            (
                mass_delta.to(env.device).unsqueeze(-1),
                com_delta.to(env.device),
            ),
            dim=-1,
        )

        controlled_priv = validate_priv_latent_values(
            controlled_observations,
            expected_mass_com=expected_mass_com,
            phase="controlled parameter test",
        )

        # ------------------------------------------------------------------
        # 6. Explicitly validate controlled ranges and ordering
        # ------------------------------------------------------------------
        torch.testing.assert_close(
            controlled_priv[:, 4:5],
            controlled_friction.to(
                env.device
            ).unsqueeze(-1),
            rtol=1.0e-5,
            atol=1.0e-6,
            msg="Controlled friction was encoded incorrectly.",
        )

        torch.testing.assert_close(
            controlled_priv[:, 5:17],
            stiffness_multiplier - 1.0,
            rtol=1.0e-5,
            atol=1.0e-6,
            msg=(
                "Stiffness offsets do not preserve "
                "FR, FL, RR, RL order."
            ),
        )

        torch.testing.assert_close(
            controlled_priv[:, 17:29],
            damping_multiplier - 1.0,
            rtol=1.0e-5,
            atol=1.0e-6,
            msg=(
                "Damping offsets do not preserve "
                "FR, FL, RR, RL order."
            ),
        )

        print("[PASS] Controlled added mass is reported in kilograms.")
        print("[PASS] Controlled CoM offsets are reported in meters.")
        print("[PASS] Controlled friction matches PhysX material values.")
        print("[PASS] Stiffness offsets preserve FR, FL, RR, RL order.")
        print("[PASS] Damping offsets preserve FR, FL, RR, RL order.")
        print("[PASS] Gain encoding is multiplier - 1.")
        print("[PASS] Official 29-D priv_latent contract validated.")

        print("\npriv_latent slice contract:")
        for name, term_slice in priv_latent_slices.items():
            print(
                f"  {name:18s}: "
                f"[{term_slice.start:02d}:{term_slice.stop:02d}]"
            )

    finally:
        env.close()

def _validate_proprio_history_contract() -> None:
    """Validate the official frame-major 10 x 53 proprio history."""

    env_cfg = Go2ExtremeParkourTeacherEnvCfg()

    # One environment for each terrain family.
    env_cfg.scene.num_envs = 5
    env_cfg.sim.device = args_cli.device

    terrain_generator_cfg = env_cfg.scene.terrain.terrain_generator
    if terrain_generator_cfg is None:
        raise RuntimeError(
            "Extreme Parkour requires a terrain generator."
        )

    terrain_generator_cfg.num_rows = 2
    terrain_generator_cfg.num_cols = 5
    terrain_generator_cfg.curriculum = True

    env_cfg.scene.terrain.max_init_terrain_level = 0

    env = ManagerBasedRLEnv(cfg=env_cfg)

    try:
        observations, _ = env.reset()

        # --------------------------------------------------------------
        # 1. ObservationManager metadata
        # --------------------------------------------------------------
        expected_term_names = ["history"]

        actual_term_names = (
            env.observation_manager.active_terms.get(
                "proprio_history"
            )
        )

        if actual_term_names != expected_term_names:
            raise RuntimeError(
                "Incorrect proprio_history term order:\n"
                f"expected={expected_term_names}\n"
                f"actual={actual_term_names}"
            )

        expected_term_dims = [(530,)]

        actual_term_dims = (
            env.observation_manager
            .group_obs_term_dim["proprio_history"]
        )

        if actual_term_dims != expected_term_dims:
            raise RuntimeError(
                "Incorrect proprio_history term dimensions:\n"
                f"expected={expected_term_dims}\n"
                f"actual={actual_term_dims}"
            )

        proprio_history = observations["proprio_history"]

        if proprio_history.shape != (env.num_envs, 530): # type:ignore
            raise RuntimeError(
                "Expected proprio_history shape "
                f"({env.num_envs}, 530), "
                f"got {tuple(proprio_history.shape)}." # type:ignore
            )

        if not torch.isfinite(proprio_history).all(): # type:ignore
            raise RuntimeError(
                "Initial proprio_history contains NaN or Inf."
            )

        print(
            "[PASS] proprio_history group shape is "
            "[num_envs, 530]."
        )
        print(
            "[PASS] proprio_history contains one "
            "flattened history term."
        )

        # Restore the unflattened temporal view:
        #
        # [num_envs, 530]
        #     ->
        # [num_envs, 10, 53]
        initial_history = proprio_history.reshape( # type:ignore
            env.num_envs,
            10,
            53,
        )

        # Helper used throughout this test. The current proprio keeps
        # its target-yaw fields, while only the history copy masks them.
        def mask_history_yaw(
            current_proprio: torch.Tensor,
        ) -> torch.Tensor:
            expected_frame = current_proprio.clone()
            expected_frame[:, 6:8] = 0.0
            return expected_frame

        initial_proprio = observations["proprio"]

        if initial_proprio.shape != (env.num_envs, 53): # type:ignore
            raise RuntimeError(
                "Expected current proprio shape "
                f"({env.num_envs}, 53), "
                f"got {tuple(initial_proprio.shape)}." # type:ignore
            )

        expected_initial_frame = mask_history_yaw(
            initial_proprio # type:ignore
        )

        # --------------------------------------------------------------
        # 2. Reset-fill behavior
        # --------------------------------------------------------------
        # On the first append after reset, IsaacLab CircularBuffer fills
        # all ten temporal positions with the same current frame.
        expected_initial_history = (
            expected_initial_frame
            .unsqueeze(1)
            .expand(-1, 10, -1)
        )

        torch.testing.assert_close(
            initial_history,
            expected_initial_history,
            rtol=1.0e-5,
            atol=1.0e-6,
            msg=(
                "The initial history was not filled with ten "
                "copies of the reset frame."
            ),
        )

        print(
            "[PASS] Reset fills all ten history slots "
            "with the current frame."
        )

        # --------------------------------------------------------------
        # 3. History-only yaw masking
        # --------------------------------------------------------------
        history_yaw = initial_history[:, :, 6:8]

        torch.testing.assert_close(
            history_yaw,
            torch.zeros_like(history_yaw),
            rtol=0.0,
            atol=0.0,
            msg=(
                "Target-yaw channels 6:8 are not zero "
                "throughout the history."
            ),
        )

        # Verify every other history channel still equals the current
        # proprio source. This catches accidental masking of the whole
        # navigation block or an off-by-one slice.
        unmasked_indices = torch.cat(
            (
                torch.arange(
                    0,
                    6,
                    device=env.device,
                ),
                torch.arange(
                    8,
                    53,
                    device=env.device,
                ),
            )
        )

        torch.testing.assert_close(
            initial_history[:, :, unmasked_indices],
            initial_proprio[
                :,
                unmasked_indices, # type:ignore
            ].unsqueeze(1).expand(-1, 10, -1),
            rtol=1.0e-5,
            atol=1.0e-6,
            msg=(
                "Non-yaw history channels do not match "
                "the current proprio frame."
            ),
        )

        print(
            "[PASS] Only full-frame channels 6:8 "
            "are masked in history."
        )

        # --------------------------------------------------------------
        # 4. Step and temporal-shift behavior
        # --------------------------------------------------------------
        # Use distinct, non-zero action values. This makes the newest
        # frame identifiable through its action slice [37:49].
        test_action_single = torch.linspace(
            -0.6,
            0.6,
            EXTREME_PARKOUR_NUM_ACTIONS,
            device=env.device,
        )

        test_actions = (
            test_action_single
            .unsqueeze(0)
            .repeat(env.num_envs, 1)
        )

        history_before_step = initial_history.clone()

        observations, _, _, _, _ = env.step(
            test_actions
        )

        stepped_history_flat = observations[
            "proprio_history"
        ]

        if stepped_history_flat.shape != ( # type:ignore
            env.num_envs,
            530,
        ):
            raise RuntimeError(
                "proprio_history shape changed after step: "
                f"{tuple(stepped_history_flat.shape)}." # type:ignore
            )

        if not torch.isfinite(stepped_history_flat).all(): # type:ignore
            raise RuntimeError(
                "Stepped proprio_history contains NaN or Inf."
            )

        stepped_history = stepped_history_flat.reshape( # type:ignore
            env.num_envs,
            10,
            53,
        )

        stepped_proprio = observations["proprio"]
        expected_latest_frame = mask_history_yaw(
            stepped_proprio # type:ignore
        )

        # The previous frames 1...9 must become the new frames 0...8.
        torch.testing.assert_close(
            stepped_history[:, 0:9, :],
            history_before_step[:, 1:10, :],
            rtol=1.0e-5,
            atol=1.0e-6,
            msg=(
                "History did not shift from old[1:10] "
                "to new[0:9]."
            ),
        )

        # The final slot must contain the newest complete 53-D frame.
        torch.testing.assert_close(
            stepped_history[:, 9, :],
            expected_latest_frame,
            rtol=1.0e-5,
            atol=1.0e-6,
            msg=(
                "The newest masked proprio frame was not "
                "written to history slot 9."
            ),
        )

        print(
            "[PASS] History shifts toward the oldest slot "
            "after one policy step."
        )
        print(
            "[PASS] The newest 53-D frame is stored "
            "in temporal slot 9."
        )

        # The newest action slice must equal this step's raw policy
        # action. It also confirms that each 53-D frame remains intact
        # after flattening and reshaping.
        torch.testing.assert_close(
            stepped_history[:, 9, 37:49],
            test_actions,
            rtol=0.0,
            atol=0.0,
            msg=(
                "Newest history action slice [37:49] "
                "does not match the raw policy action."
            ),
        )

        torch.testing.assert_close(
            stepped_history[:, 9, 6:8],
            torch.zeros_like(
                stepped_history[:, 9, 6:8]
            ),
            rtol=0.0,
            atol=0.0,
            msg=(
                "Newest history target-yaw channels "
                "were not masked."
            ),
        )

        print(
            "[PASS] Newest history action slice "
            "[37:49] matches the raw action."
        )
        print(
            "[PASS] Newest history yaw slice "
            "[6:8] remains masked."
        )

        # --------------------------------------------------------------
        # 5. Reset after the history has changed
        # --------------------------------------------------------------
        reset_observations, _ = env.reset()

        reset_proprio = reset_observations["proprio"]

        reset_history = reset_observations[
            "proprio_history"
        ].reshape( # type:ignore
            env.num_envs,
            10,
            53,
        )

        expected_reset_frame = mask_history_yaw(
            reset_proprio # type:ignore
        )

        expected_reset_history = (
            expected_reset_frame
            .unsqueeze(1)
            .expand(-1, 10, -1)
        )

        torch.testing.assert_close(
            reset_history,
            expected_reset_history,
            rtol=1.0e-5,
            atol=1.0e-6,
            msg=(
                "Reset did not clear the old history and "
                "refill all slots with the new reset frame."
            ),
        )

        print(
            "[PASS] Episode reset removes history "
            "from the previous episode."
        )
        print(
            "[PASS] Reset refills all ten slots "
            "with the new reset frame."
        )

        # --------------------------------------------------------------
        # 6. Complete 753-D teacher/critic observation contract
        # --------------------------------------------------------------
        expected_group_dims = {
            "proprio": 53,
            "terrain_scan": 132,
            "priv_explicit": 9,
            "priv_latent": 29,
            "proprio_history": 530,
        }

        for group_name, expected_dim in (
            expected_group_dims.items()
        ):
            group_observation = reset_observations[
                group_name
            ] # type:ignore

            expected_shape = (
                env.num_envs,
                expected_dim,
            )

            if group_observation.shape != expected_shape: # type:ignore
                raise RuntimeError(
                    f"Observation group '{group_name}' "
                    f"expected shape {expected_shape}, "
                    f"got {tuple(group_observation.shape)}." # type:ignore
                )

            if not torch.isfinite(
                group_observation # type:ignore
            ).all():
                raise RuntimeError(
                    f"Observation group '{group_name}' "
                    "contains NaN or Inf."
                )

        complete_observation = torch.cat( # type:ignore
            (
                reset_observations["proprio"], # type:ignore
                reset_observations["terrain_scan"],
                reset_observations["priv_explicit"],
                reset_observations["priv_latent"],
                reset_observations["proprio_history"],
            ),
            dim=-1,
        )

        if complete_observation.shape != (
            env.num_envs,
            753,
        ):
            raise RuntimeError(
                "Expected complete teacher observation shape "
                f"({env.num_envs}, 753), "
                f"got {tuple(complete_observation.shape)}."
            )

        if not torch.isfinite(
            complete_observation
        ).all():
            raise RuntimeError(
                "Complete 753-D observation contains NaN or Inf."
            )

        print(
            "[PASS] Observation dimensions are "
            "53 + 132 + 9 + 29 + 530 = 753."
        )
        print(
            "[PASS] Official 530-D proprio-history "
            "contract validated."
        )

    finally:
        env.close()

def _validate_navigation_rewards() -> None:
    """Validate waypoint velocity/yaw rewards with controlled robot states."""

    env_cfg = Go2ExtremeParkourTeacherEnvCfg()

    # Six environments represent six controlled test cases.
    env_cfg.scene.num_envs = 6
    env_cfg.sim.device = args_cli.device

    terrain_generator_cfg = env_cfg.scene.terrain.terrain_generator
    if terrain_generator_cfg is None:
        raise RuntimeError(
            "Extreme Parkour requires a terrain generator."
        )

    terrain_generator_cfg.num_rows = 2
    terrain_generator_cfg.num_cols = 5
    terrain_generator_cfg.curriculum = True

    env_cfg.scene.terrain.max_init_terrain_level = 0

    env = ManagerBasedRLEnv(cfg=env_cfg)

    try:
        env.reset()

        # --------------------------------------------------------------
        # 1. RewardManager registration contract
        # --------------------------------------------------------------
        expected_term_names = [
            "tracking_goal_vel",
            "tracking_yaw",
        ]

        actual_term_names = env.reward_manager.active_terms

        if actual_term_names != expected_term_names:
            raise RuntimeError(
                "Incorrect reward term order:\n"
                f"expected={expected_term_names}\n"
                f"actual={actual_term_names}"
            )

        goal_vel_cfg = env.reward_manager.get_term_cfg(
            "tracking_goal_vel"
        )
        tracking_yaw_cfg = env.reward_manager.get_term_cfg(
            "tracking_yaw"
        )

        if not math.isclose(
            goal_vel_cfg.weight,
            1.5,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                "tracking_goal_vel weight must be 1.5, "
                f"got {goal_vel_cfg.weight}."
            )

        if not math.isclose(
            tracking_yaw_cfg.weight,
            0.5,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                "tracking_yaw weight must be 0.5, "
                f"got {tracking_yaw_cfg.weight}."
            )

        print("[PASS] Navigation reward terms are registered.")
        print("[PASS] Reward weights are 1.5 and 0.5.")

        robot: Articulation = env.scene["robot"]
        command_term = env.command_manager.get_term(
            "waypoint"
        )

        def evaluate_raw_reward(
            term_name: str,
        ) -> torch.Tensor:
            """Call the configured reward term without weight or dt."""

            term_cfg = env.reward_manager.get_term_cfg(
                term_name
            )

            value = term_cfg.func(
                env,
                **term_cfg.params,
            )

            if value.shape != (env.num_envs,):
                raise RuntimeError(
                    f"Reward '{term_name}' expected shape "
                    f"({env.num_envs},), got {tuple(value.shape)}."
                )

            if not torch.isfinite(value).all():
                raise RuntimeError(
                    f"Reward '{term_name}' contains NaN or Inf."
                )

            return value

        # --------------------------------------------------------------
        # 2. Velocity projection tests
        # --------------------------------------------------------------
        # Place every waypoint two meters along world +x from the robot.
        # Waypoint z is irrelevant to this two-dimensional reward.
        root_xy_w = robot.data.root_pos_w[:, :2].clone()

        command_term.current_waypoint_w[:, :2] = (  # type: ignore
            root_xy_w
        )
        command_term.current_waypoint_w[:, 0] += 2.0  # type: ignore

        target_speed = 0.5

        command_term.forward_speed.fill_(target_speed)  # type: ignore

        # Six controlled world-frame velocities:
        #
        # 0: away from target
        # 1: stopped
        # 2: half target speed
        # 3: target speed
        # 4: faster than target
        # 5: purely lateral
        controlled_velocity_xy = torch.tensor(
            [
                [-0.50, 0.00],
                [0.00, 0.00],
                [0.25, 0.00],
                [0.50, 0.00],
                [1.00, 0.00],
                [0.00, 0.50],
            ],
            device=env.device,
            dtype=torch.float32,
        )

        root_velocity_w = torch.zeros(
            env.num_envs,
            6,
            device=env.device,
        )
        root_velocity_w[:, :2] = controlled_velocity_xy

        # This API updates both PhysX and the articulation's internal
        # world-frame velocity buffers.
        robot.write_root_velocity_to_sim(
            root_velocity_w
        )

        actual_goal_vel_reward = evaluate_raw_reward(
            "tracking_goal_vel"
        )

        # Reproduce the official formula independently.
        target_pos_rel_w = (
            command_term.current_waypoint_w[:, :2]  # type: ignore
            - robot.data.root_pos_w[:, :2]
        )

        target_norm = torch.linalg.vector_norm(
            target_pos_rel_w,
            dim=-1,
            keepdim=True,
        )

        expected_target_direction_w = (
            target_pos_rel_w
            / (target_norm + 1.0e-5)
        )

        expected_projected_velocity = torch.sum(
            expected_target_direction_w
            * controlled_velocity_xy,
            dim=-1,
        )

        speed_tensor = torch.full(
            (env.num_envs,),
            target_speed,
            device=env.device,
        )

        expected_goal_vel_reward = (
            torch.minimum(
                expected_projected_velocity,
                speed_tensor,
            )
            / (speed_tensor + 1.0e-5)
        )

        torch.testing.assert_close(
            actual_goal_vel_reward,
            expected_goal_vel_reward,
            rtol=1.0e-6,
            atol=1.0e-6,
            msg=(
                "tracking_goal_vel does not match the official "
                "direction-dot-velocity formula."
            ),
        )

        if actual_goal_vel_reward[0] >= 0.0:
            raise RuntimeError(
                "Moving away from the waypoint must produce "
                "a negative goal-velocity reward."
            )

        if actual_goal_vel_reward[1] != 0.0:
            raise RuntimeError(
                "Zero velocity must produce zero "
                "goal-velocity reward."
            )

        if actual_goal_vel_reward[2] <= 0.0:
            raise RuntimeError(
                "Moving toward the waypoint must produce "
                "a positive goal-velocity reward."
            )

        if actual_goal_vel_reward[5] != 0.0:
            raise RuntimeError(
                "Pure lateral velocity must produce zero "
                "goal-direction projection."
            )

        maximum_reward = (
            target_speed
            / (target_speed + 1.0e-5)
        )

        torch.testing.assert_close(
            actual_goal_vel_reward[4],
            torch.tensor(
                maximum_reward,
                device=env.device,
            ),
            rtol=1.0e-6,
            atol=1.0e-6,
            msg=(
                "Velocity above target speed was not capped."
            ),
        )

        print("[PASS] Moving toward the waypoint is rewarded.")
        print("[PASS] Moving away from the waypoint is negative.")
        print("[PASS] Lateral velocity has zero projection.")
        print("[PASS] Excess forward velocity is capped.")

        # --------------------------------------------------------------
        # 3. Controlled yaw tests
        # --------------------------------------------------------------
        # Set every robot to identity orientation: yaw = 0.
        root_pose_w = robot.data.root_state_w[:, :7].clone()
        root_pose_w[:, 3:7] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            device=env.device,
        )

        robot.write_root_pose_to_sim(
            root_pose_w
        )

        actual_robot_yaw = robot.data.heading_w

        torch.testing.assert_close(
            actual_robot_yaw,
            torch.zeros_like(actual_robot_yaw),
            rtol=0.0,
            atol=1.0e-6,
            msg="Controlled robot yaw is not zero.",
        )

        yaw_errors = torch.tensor(
            [
                0.0,
                math.pi / 6.0,
                math.pi / 2.0,
                math.pi,
                -math.pi / 2.0,
                -math.pi,
            ],
            device=env.device,
            dtype=torch.float32,
        )

        # Construct one waypoint direction for each desired yaw error.
        target_directions_w = torch.stack(
            (
                torch.cos(yaw_errors),
                torch.sin(yaw_errors),
            ),
            dim=-1,
        )

        command_term.current_waypoint_w[:, :2] = (  # type: ignore
            robot.data.root_pos_w[:, :2]
            + 2.0 * target_directions_w
        )

        actual_yaw_reward = evaluate_raw_reward(
            "tracking_yaw"
        )

        expected_yaw_reward = torch.exp(
            -torch.abs(yaw_errors)
        )

        torch.testing.assert_close(
            actual_yaw_reward,
            expected_yaw_reward,
            rtol=1.0e-5,
            atol=1.0e-6,
            msg=(
                "tracking_yaw does not equal "
                "exp(-abs(target_yaw - robot_yaw))."
            ),
        )

        torch.testing.assert_close(
            actual_yaw_reward[0],
            torch.tensor(
                1.0,
                device=env.device,
            ),
            rtol=0.0,
            atol=1.0e-6,
            msg="Zero yaw error must produce reward 1.",
        )

        if not (
            actual_yaw_reward[0]
            > actual_yaw_reward[1]
            > actual_yaw_reward[2]
            > actual_yaw_reward[3]
        ):
            raise RuntimeError(
                "Yaw reward must decrease monotonically "
                "as absolute yaw error increases."
            )

        torch.testing.assert_close(
            actual_yaw_reward[2],
            actual_yaw_reward[4],
            rtol=1.0e-5,
            atol=1.0e-6,
            msg=(
                "Positive and negative yaw errors with equal "
                "magnitude must receive equal reward."
            ),
        )

        print("[PASS] Zero yaw error produces reward 1.")
        print("[PASS] Yaw reward decreases with absolute error.")
        print("[PASS] Positive/negative equal errors are symmetric.")

        # --------------------------------------------------------------
        # 4. RewardManager weight and dt integration
        # --------------------------------------------------------------
        raw_goal_vel = evaluate_raw_reward(
            "tracking_goal_vel"
        )
        raw_yaw = evaluate_raw_reward(
            "tracking_yaw"
        )

        expected_total_reward = (
            1.5 * raw_goal_vel
            + 0.5 * raw_yaw
        ) * env.step_dt

        manager_total_reward = env.reward_manager.compute(
            dt=env.step_dt
        )

        torch.testing.assert_close(
            manager_total_reward,
            expected_total_reward,
            rtol=1.0e-6,
            atol=1.0e-6,
            msg=(
                "RewardManager did not apply official weights "
                "and policy dt correctly."
            ),
        )

        if not math.isclose(
            env.step_dt,
            0.02,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError(
                f"Expected policy dt 0.02 s, got {env.step_dt}."
            )

        print("[PASS] RewardManager applies weights 1.5 and 0.5.")
        print("[PASS] RewardManager multiplies the sum by dt=0.02.")
        print("[PASS] Navigation reward contract validated.")

        print("\nControlled velocity rewards:")
        for index in range(env.num_envs):
            print(
                f"  case {index}: "
                f"velocity={controlled_velocity_xy[index].tolist()}, "
                f"reward={actual_goal_vel_reward[index].item():.6f}"
            )

        print("\nControlled yaw rewards:")
        for index in range(env.num_envs):
            print(
                f"  case {index}: "
                f"yaw_error={yaw_errors[index].item():+.6f}, "
                f"reward={actual_yaw_reward[index].item():.6f}"
            )

    finally:
        env.close()

def _validate_body_regularization_rewards() -> None:
    """Validate body-motion and orientation regularization rewards."""

    env_cfg = Go2ExtremeParkourTeacherEnvCfg()

    env_cfg.scene.num_envs = 6
    env_cfg.sim.device = args_cli.device

    terrain_generator_cfg = env_cfg.scene.terrain.terrain_generator
    if terrain_generator_cfg is None:
        raise RuntimeError(
            "Extreme Parkour requires a terrain generator."
        )

    terrain_generator_cfg.num_rows = 2
    terrain_generator_cfg.num_cols = 5
    terrain_generator_cfg.curriculum = True

    env_cfg.scene.terrain.max_init_terrain_level = 0

    env = ManagerBasedRLEnv(cfg=env_cfg)

    try:
        env.reset()

        robot: Articulation = env.scene["robot"]

        command_term = env.command_manager.get_term(
            "waypoint"
        )

        # --------------------------------------------------------------
        # 1. RewardManager metadata
        # --------------------------------------------------------------
        expected_term_names = [
            "tracking_goal_vel",
            "tracking_yaw",
            "lin_vel_z",
            "ang_vel_xy",
            "orientation",
        ]

        actual_term_names = env.reward_manager.active_terms

        if actual_term_names != expected_term_names:
            raise RuntimeError(
                "Incorrect reward term order:\n"
                f"expected={expected_term_names}\n"
                f"actual={actual_term_names}"
            )

        expected_weights = {
            "tracking_goal_vel": 1.5,
            "tracking_yaw": 0.5,
            "lin_vel_z": -1.0,
            "ang_vel_xy": -0.05,
            "orientation": -1.0,
        }

        for term_name, expected_weight in (
            expected_weights.items()
        ):
            actual_weight = (
                env.reward_manager
                .get_term_cfg(term_name)
                .weight
            )

            if not math.isclose(
                actual_weight,
                expected_weight,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(
                    f"Reward '{term_name}' expected weight "
                    f"{expected_weight}, got {actual_weight}."
                )

        print("[PASS] Five reward terms are registered.")
        print("[PASS] Body regularization weights are correct.")

        def evaluate_raw_reward(
            term_name: str,
        ) -> torch.Tensor:
            """Evaluate one configured term without weight or dt."""

            term_cfg = env.reward_manager.get_term_cfg(
                term_name
            )

            value = term_cfg.func(
                env,
                **term_cfg.params,
            )

            if value.shape != (env.num_envs,):
                raise RuntimeError(
                    f"Reward '{term_name}' expected shape "
                    f"({env.num_envs},), got {tuple(value.shape)}."
                )

            if not torch.isfinite(value).all():
                raise RuntimeError(
                    f"Reward '{term_name}' contains NaN or Inf."
                )

            return value

        # --------------------------------------------------------------
        # 2. Controlled terrain classes
        # --------------------------------------------------------------
        flat_class = int(
            command_term.cfg.parkour_flat_terrain_class  # type: ignore
        )

        if flat_class != 2:
            raise RuntimeError(
                "This project expects local terrain class 2 "
                f"to represent flat terrain, got {flat_class}."
            )

        # Environments 0, 2 and 4 are flat.
        # Environments 1, 3 and 5 are non-flat parkour terrain.
        controlled_terrain_classes = torch.tensor(
            [2, 0, 2, 1, 2, 4],
            device=env.device,
            dtype=torch.long,
        )

        command_term.terrain_class.copy_(  # type: ignore
            controlled_terrain_classes
        )

        is_flat = (
            controlled_terrain_classes == flat_class
        )

        print("[PASS] Controlled flat/non-flat classes installed.")

        # --------------------------------------------------------------
        # 3. Set identity base orientation
        # --------------------------------------------------------------
        # With identity orientation, body-frame and world-frame velocity
        # components are identical. This makes expected values analytic.
        identity_root_pose = (
            robot.data.root_state_w[:, :7].clone()
        )

        identity_root_pose[:, 3:7] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            device=env.device,
        )

        robot.write_root_pose_to_sim(
            identity_root_pose
        )

        torch.testing.assert_close(
            robot.data.heading_w,
            torch.zeros_like(
                robot.data.heading_w
            ),
            rtol=0.0,
            atol=1.0e-6,
            msg="Controlled identity pose must have yaw zero.",
        )

        # --------------------------------------------------------------
        # 4. Vertical linear velocity penalty
        # --------------------------------------------------------------
        controlled_lin_vel_z = torch.tensor(
            [0.0, 1.0, -2.0, 1.5, 0.5, -1.0],
            device=env.device,
        )

        controlled_ang_vel_xy = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 2.0],
                [1.0, 2.0],
                [-0.5, 0.5],
                [3.0, 4.0],
            ],
            device=env.device,
        )

        root_velocity_w = torch.zeros(
            env.num_envs,
            6,
            device=env.device,
        )

        # Columns 0:3 are world linear velocity.
        root_velocity_w[:, 2] = controlled_lin_vel_z

        # Columns 3:6 are world angular velocity.
        root_velocity_w[:, 3:5] = (
            controlled_ang_vel_xy
        )

        robot.write_root_velocity_to_sim(
            root_velocity_w
        )

        torch.testing.assert_close(
            robot.data.root_lin_vel_b[:, 2],
            controlled_lin_vel_z,
            rtol=1.0e-6,
            atol=1.0e-6,
            msg=(
                "Identity-pose body vertical velocity "
                "does not match the controlled value."
            ),
        )

        actual_lin_vel_z = evaluate_raw_reward(
            "lin_vel_z"
        )

        terrain_scale = torch.where(
            is_flat,
            torch.ones(
                env.num_envs,
                device=env.device,
            ),
            torch.full(
                (env.num_envs,),
                0.5,
                device=env.device,
            ),
        )

        expected_lin_vel_z = (
            torch.square(controlled_lin_vel_z)
            * terrain_scale
        )

        torch.testing.assert_close(
            actual_lin_vel_z,
            expected_lin_vel_z,
            rtol=1.0e-6,
            atol=1.0e-6,
            msg=(
                "lin_vel_z does not implement the official "
                "flat/non-flat scaling."
            ),
        )

        # Cases 1 and 5 are non-flat with |vz| = 1.
        torch.testing.assert_close(
            actual_lin_vel_z[[1, 5]],
            torch.tensor(
                [0.5, 0.5],
                device=env.device,
            ),
            rtol=0.0,
            atol=1.0e-6,
        )

        print("[PASS] lin_vel_z equals body-frame vz squared.")
        print("[PASS] Non-flat lin_vel_z penalty is halved.")
        print("[PASS] Flat lin_vel_z penalty keeps full scale.")

        # --------------------------------------------------------------
        # 5. Roll/pitch angular velocity penalty
        # --------------------------------------------------------------
        torch.testing.assert_close(
            robot.data.root_ang_vel_b[:, :2],
            controlled_ang_vel_xy,
            rtol=1.0e-6,
            atol=1.0e-6,
            msg=(
                "Identity-pose body angular velocity "
                "does not match controlled values."
            ),
        )

        actual_ang_vel_xy = evaluate_raw_reward(
            "ang_vel_xy"
        )

        expected_ang_vel_xy = torch.sum(
            torch.square(controlled_ang_vel_xy),
            dim=-1,
        )

        torch.testing.assert_close(
            actual_ang_vel_xy,
            expected_ang_vel_xy,
            rtol=1.0e-6,
            atol=1.0e-6,
            msg=(
                "ang_vel_xy does not equal "
                "angular_velocity_x^2 + angular_velocity_y^2."
            ),
        )

        expected_ang_vel_values = torch.tensor(
            [0.0, 1.0, 4.0, 5.0, 0.5, 25.0],
            device=env.device,
        )

        torch.testing.assert_close(
            actual_ang_vel_xy,
            expected_ang_vel_values,
            rtol=1.0e-6,
            atol=1.0e-6,
        )

        print("[PASS] ang_vel_xy equals wx^2 + wy^2.")
        print("[PASS] ang_vel_xy is independent of terrain class.")

        # --------------------------------------------------------------
        # 6. Orientation penalty
        # --------------------------------------------------------------
        # Use pure roll rotations. For roll theta:
        #
        # projected_gravity_xy squared sum = sin(theta)^2.
        controlled_roll = torch.tensor(
            [
                0.0,
                math.pi / 6.0,
                math.pi / 6.0,
                math.pi / 4.0,
                math.pi / 4.0,
                math.pi / 3.0,
            ],
            device=env.device,
        )

        zeros = torch.zeros_like(
            controlled_roll
        )

        controlled_quat_w = (
            math_utils.quat_from_euler_xyz(
                controlled_roll,
                zeros,
                zeros,
            )
        )

        tilted_root_pose = (
            robot.data.root_state_w[:, :7].clone()
        )

        tilted_root_pose[:, 3:7] = controlled_quat_w

        robot.write_root_pose_to_sim(
            tilted_root_pose
        )

        actual_orientation = evaluate_raw_reward(
            "orientation"
        )

        expected_unmasked_orientation = torch.square(
            torch.sin(controlled_roll)
        )

        expected_orientation = (
            expected_unmasked_orientation
            * is_flat.float()
        )

        torch.testing.assert_close(
            actual_orientation,
            expected_orientation,
            rtol=1.0e-5,
            atol=1.0e-6,
            msg=(
                "orientation does not equal projected gravity "
                "xy error gated by the flat terrain class."
            ),
        )

        # Environment 1 and 2 use the same roll angle.
        # Only environment 2 is flat.
        torch.testing.assert_close(
            actual_orientation[1],
            torch.tensor(
                0.0,
                device=env.device,
            ),
            rtol=0.0,
            atol=1.0e-6,
            msg=(
                "Orientation penalty must be zero "
                "on non-flat terrain."
            ),
        )

        torch.testing.assert_close(
            actual_orientation[2],
            torch.tensor(
                0.25,
                device=env.device,
            ),
            rtol=1.0e-5,
            atol=1.0e-6,
            msg=(
                "A flat-terrain roll of pi/6 must "
                "produce sin(pi/6)^2 = 0.25."
            ),
        )

        # Environments 3 and 4 both use roll pi/4.
        torch.testing.assert_close(
            actual_orientation[3],
            torch.tensor(
                0.0,
                device=env.device,
            ),
            rtol=0.0,
            atol=1.0e-6,
        )

        torch.testing.assert_close(
            actual_orientation[4],
            torch.tensor(
                0.5,
                device=env.device,
            ),
            rtol=1.0e-5,
            atol=1.0e-6,
        )

        print("[PASS] Upright orientation penalty is zero.")
        print("[PASS] Flat orientation penalty is sin(roll)^2.")
        print("[PASS] Non-flat orientation penalty is disabled.")

        # --------------------------------------------------------------
        # 7. Verify sign, weight and dt through RewardManager
        # --------------------------------------------------------------
        raw_rewards = {
            term_name: evaluate_raw_reward(term_name)
            for term_name in expected_term_names
        }

        expected_total_reward = (
            1.5 * raw_rewards["tracking_goal_vel"]
            + 0.5 * raw_rewards["tracking_yaw"]
            - 1.0 * raw_rewards["lin_vel_z"]
            - 0.05 * raw_rewards["ang_vel_xy"]
            - 1.0 * raw_rewards["orientation"]
        ) * env.step_dt

        actual_total_reward = env.reward_manager.compute(
            dt=env.step_dt
        )

        torch.testing.assert_close(
            actual_total_reward,
            expected_total_reward,
            rtol=1.0e-5,
            atol=1.0e-6,
            msg=(
                "RewardManager did not apply body penalty "
                "weights and dt correctly."
            ),
        )

        # The raw penalty terms must remain non-negative.
        for term_name in (
            "lin_vel_z",
            "ang_vel_xy",
            "orientation",
        ):
            if torch.any(raw_rewards[term_name] < 0.0):
                raise RuntimeError(
                    f"Raw penalty '{term_name}' must be non-negative."
                )

        print("[PASS] Raw regularization terms are non-negative.")
        print("[PASS] Negative weights convert them into penalties.")
        print("[PASS] RewardManager applies policy dt exactly once.")
        print("[PASS] Body regularization reward contract validated.")

        print("\nBody regularization cases:")
        for index in range(env.num_envs):
            print(
                f"  env {index}: "
                f"class={int(controlled_terrain_classes[index])}, "
                f"vz={controlled_lin_vel_z[index].item():+.2f}, "
                f"ang_xy={actual_ang_vel_xy[index].item():.3f}, "
                f"roll={controlled_roll[index].item():+.3f}, "
                f"orientation={actual_orientation[index].item():.3f}"
            )

    finally:
        env.close()

def _validate_joint_regularization_rewards() -> None:
    """Validate stateful joint/action/torque regularization terms."""

    env_cfg = Go2ExtremeParkourTeacherEnvCfg()

    env_cfg.scene.num_envs = 6
    env_cfg.sim.device = args_cli.device

    terrain_generator_cfg = env_cfg.scene.terrain.terrain_generator
    if terrain_generator_cfg is None:
        raise RuntimeError(
            "Extreme Parkour requires a terrain generator."
        )

    terrain_generator_cfg.num_rows = 2
    terrain_generator_cfg.num_cols = 5
    terrain_generator_cfg.curriculum = True
    env_cfg.scene.terrain.max_init_terrain_level = 0

    env = ManagerBasedRLEnv(cfg=env_cfg)

    try:
        env.reset()

        robot: Articulation = env.scene["robot"]

        # --------------------------------------------------------------
        # 1. Reward registration and weights
        # --------------------------------------------------------------
        expected_term_names = [
            "tracking_goal_vel",
            "tracking_yaw",
            "lin_vel_z",
            "ang_vel_xy",
            "orientation",
            "dof_acc",
            "action_rate",
            "delta_torques",
            "torques",
        ]

        if env.reward_manager.active_terms != expected_term_names:
            raise RuntimeError(
                "Incorrect reward term order:\n"
                f"expected={expected_term_names}\n"
                f"actual={env.reward_manager.active_terms}"
            )

        expected_weights = {
            "dof_acc": -2.5e-7,
            "action_rate": -0.1,
            "delta_torques": -1.0e-7,
            "torques": -1.0e-5,
        }

        for term_name, expected_weight in (
            expected_weights.items()
        ):
            actual_weight = (
                env.reward_manager
                .get_term_cfg(term_name)
                .weight
            )

            if not math.isclose(
                actual_weight,
                expected_weight,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(
                    f"Reward '{term_name}' expected weight "
                    f"{expected_weight}, got {actual_weight}."
                )

        if not math.isclose(
            env.step_dt,
            0.02,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError(
                f"Expected policy step_dt 0.02, got {env.step_dt}."
            )

        print("[PASS] Nine reward terms are registered.")
        print("[PASS] Joint regularization weights are correct.")
        print("[PASS] Policy step_dt is 0.02 s.")

        def evaluate_term(
            term_name: str,
        ) -> torch.Tensor:
            """Call a configured reward term without weight or dt."""

            term_cfg = env.reward_manager.get_term_cfg(
                term_name
            )

            result = term_cfg.func(
                env,
                **term_cfg.params,
            )

            if result.shape != (env.num_envs,):
                raise RuntimeError(
                    f"Reward '{term_name}' returned shape "
                    f"{tuple(result.shape)}."
                )

            if not torch.isfinite(result).all():
                raise RuntimeError(
                    f"Reward '{term_name}' contains NaN or Inf."
                )

            return result

        dof_acc_cfg = env.reward_manager.get_term_cfg(
            "dof_acc"
        )
        delta_torque_cfg = env.reward_manager.get_term_cfg(
            "delta_torques"
        )

        joint_ids_cfg = dof_acc_cfg.params["asset_cfg"]

        if isinstance(joint_ids_cfg.joint_ids, slice):
            raise RuntimeError(
                "dof_acc must resolve 12 explicit joint IDs."
            )

        joint_ids = list(
            joint_ids_cfg.joint_ids
        )

        if len(joint_ids) != 12:
            raise RuntimeError(
                f"Expected 12 joint IDs, got {len(joint_ids)}."
            )

        # Ensure the test begins from zero history.
        dof_acc_cfg.func.reset()
        delta_torque_cfg.func.reset()

        # --------------------------------------------------------------
        # 2. First and second joint-velocity states
        # --------------------------------------------------------------
        joint_pattern = torch.linspace(
            -0.6,
            0.6,
            12,
            device=env.device,
        )

        env_scale = torch.arange(
            1,
            env.num_envs + 1,
            device=env.device,
            dtype=torch.float32,
        ).unsqueeze(-1)

        joint_vel_1 = (
            0.2 * env_scale * joint_pattern.unsqueeze(0)
        )

        robot.write_joint_velocity_to_sim(
            joint_vel_1,
            joint_ids=joint_ids,
        )

        dof_acc_1 = evaluate_term("dof_acc")

        # History was explicitly reset to zero.
        expected_dof_acc_1 = torch.sum(
            torch.square(
                joint_vel_1 / env.step_dt
            ),
            dim=-1,
        )

        torch.testing.assert_close(
            dof_acc_1,
            expected_dof_acc_1,
            rtol=1.0e-5,
            atol=1.0e-5,
            msg=(
                "First dof_acc call must compare current "
                "joint velocity against zero history."
            ),
        )

        joint_vel_2 = (
            joint_vel_1
            + 0.04 * env_scale
        )

        robot.write_joint_velocity_to_sim(
            joint_vel_2,
            joint_ids=joint_ids,
        )

        dof_acc_2 = evaluate_term("dof_acc")

        expected_dof_acc_2 = torch.sum(
            torch.square(
                (
                    joint_vel_2
                    - joint_vel_1
                ) / env.step_dt
            ),
            dim=-1,
        )

        torch.testing.assert_close(
            dof_acc_2,
            expected_dof_acc_2,
            rtol=1.0e-5,
            atol=1.0e-5,
            msg=(
                "Second dof_acc call must compare against "
                "the preceding policy-step joint velocity."
            ),
        )

        print("[PASS] First dof_acc uses zero reset history.")
        print("[PASS] Second dof_acc uses previous policy-step velocity.")
        print("[PASS] dof_acc divides by step_dt=0.02.")

        # --------------------------------------------------------------
        # 3. Action L2-norm tests
        # --------------------------------------------------------------
        env.action_manager.reset()

        action_pattern = torch.linspace(
            -0.5,
            0.5,
            12,
            device=env.device,
        )

        action_1 = (
            env_scale * action_pattern.unsqueeze(0)
            / float(env.num_envs)
        )

        # Public API: shifts current action to prev_action, then stores
        # action_1 as the current action.
        env.action_manager.process_action(
            action_1
        )

        action_rate_1 = evaluate_term(
            "action_rate"
        )

        expected_action_rate_1 = (
            torch.linalg.vector_norm(
                action_1,
                dim=-1,
            )
        )

        torch.testing.assert_close(
            action_rate_1,
            expected_action_rate_1,
            rtol=1.0e-6,
            atol=1.0e-6,
            msg=(
                "First action_rate must be the L2 norm "
                "between action_1 and zero."
            ),
        )

        action_2 = -0.5 * action_1

        env.action_manager.process_action(
            action_2
        )

        action_rate_2 = evaluate_term(
            "action_rate"
        )

        expected_action_rate_2 = (
            torch.linalg.vector_norm(
                action_2 - action_1,
                dim=-1,
            )
        )

        torch.testing.assert_close(
            action_rate_2,
            expected_action_rate_2,
            rtol=1.0e-6,
            atol=1.0e-6,
            msg=(
                "Second action_rate must be the L2 norm "
                "between action_2 and action_1."
            ),
        )

        # Explicitly prove that this is not the squared L2 term.
        squared_difference = torch.sum(
            torch.square(
                action_2 - action_1
            ),
            dim=-1,
        )

        if torch.allclose(
            action_rate_2,
            squared_difference,
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise RuntimeError(
                "action_rate accidentally matches squared L2; "
                "the official reward requires the L2 norm."
            )

        print("[PASS] action_rate uses current and previous actions.")
        print("[PASS] action_rate is an L2 norm, not squared L2.")

        # --------------------------------------------------------------
        # 4. Torque magnitude and torque-change tests
        # --------------------------------------------------------------
        torque_pattern = torch.linspace(
            -4.0,
            4.0,
            12,
            device=env.device,
        )

        torque_1 = (
            env_scale * torque_pattern.unsqueeze(0)
            / float(env.num_envs)
        )

        # Diagnostic-only controlled buffer assignment. Production reward
        # reads this buffer after the actuator/physics update.
        robot.data.applied_torque[
            :,
            joint_ids,
        ] = torque_1

        delta_torque_1 = evaluate_term(
            "delta_torques"
        )

        torque_l2_1 = evaluate_term(
            "torques"
        )

        expected_delta_torque_1 = torch.sum(
            torch.square(torque_1),
            dim=-1,
        )

        expected_torque_l2_1 = torch.sum(
            torch.square(torque_1),
            dim=-1,
        )

        torch.testing.assert_close(
            delta_torque_1,
            expected_delta_torque_1,
            rtol=1.0e-6,
            atol=1.0e-6,
            msg=(
                "First delta_torques call must compare "
                "torque_1 against zero history."
            ),
        )

        torch.testing.assert_close(
            torque_l2_1,
            expected_torque_l2_1,
            rtol=1.0e-6,
            atol=1.0e-6,
            msg="torques must equal sum(torque^2).",
        )

        torque_2 = (
            -0.25 * torque_1
            + 0.3 * env_scale
        )

        robot.data.applied_torque[
            :,
            joint_ids,
        ] = torque_2

        delta_torque_2 = evaluate_term(
            "delta_torques"
        )

        torque_l2_2 = evaluate_term(
            "torques"
        )

        expected_delta_torque_2 = torch.sum(
            torch.square(
                torque_2 - torque_1
            ),
            dim=-1,
        )

        expected_torque_l2_2 = torch.sum(
            torch.square(torque_2),
            dim=-1,
        )

        torch.testing.assert_close(
            delta_torque_2,
            expected_delta_torque_2,
            rtol=1.0e-6,
            atol=1.0e-6,
            msg=(
                "Second delta_torques call must compare "
                "torque_2 against torque_1."
            ),
        )

        torch.testing.assert_close(
            torque_l2_2,
            expected_torque_l2_2,
            rtol=1.0e-6,
            atol=1.0e-6,
            msg="Second torques value is incorrect.",
        )

        print("[PASS] First torque change uses zero reset history.")
        print("[PASS] Second torque change uses previous torque.")
        print("[PASS] torques equals sum(applied_torque^2).")

        # --------------------------------------------------------------
        # 5. Partial-environment reset
        # --------------------------------------------------------------
        reset_env_ids = torch.tensor(
            [0, 2, 4],
            device=env.device,
            dtype=torch.long,
        )

        # This exercises RewardManager's real reset path. It resets both
        # stateful reward terms only for the selected environments.
        env.reward_manager.reset(
            reset_env_ids # type:ignore
        )

        joint_vel_3 = (
            joint_vel_2
            + 0.03
        )

        robot.write_joint_velocity_to_sim(
            joint_vel_3,
            joint_ids=joint_ids,
        )

        dof_acc_3 = evaluate_term(
            "dof_acc"
        )

        expected_previous_joint_vel = (
            joint_vel_2.clone()
        )
        expected_previous_joint_vel[
            reset_env_ids
        ] = 0.0

        expected_dof_acc_3 = torch.sum(
            torch.square(
                (
                    joint_vel_3
                    - expected_previous_joint_vel
                ) / env.step_dt
            ),
            dim=-1,
        )

        torch.testing.assert_close(
            dof_acc_3,
            expected_dof_acc_3,
            rtol=1.0e-5,
            atol=1.0e-5,
            msg=(
                "Partial reset did not clear dof_acc history "
                "only for selected environments."
            ),
        )

        torque_3 = torque_2 + 0.2

        robot.data.applied_torque[
            :,
            joint_ids,
        ] = torque_3

        delta_torque_3 = evaluate_term(
            "delta_torques"
        )

        expected_previous_torque = torque_2.clone()
        expected_previous_torque[
            reset_env_ids
        ] = 0.0

        expected_delta_torque_3 = torch.sum(
            torch.square(
                torque_3
                - expected_previous_torque
            ),
            dim=-1,
        )

        torch.testing.assert_close(
            delta_torque_3,
            expected_delta_torque_3,
            rtol=1.0e-6,
            atol=1.0e-6,
            msg=(
                "Partial reset did not clear torque history "
                "only for selected environments."
            ),
        )

        print("[PASS] Partial reset clears selected dof history.")
        print("[PASS] Partial reset clears selected torque history.")
        print("[PASS] Non-reset environments preserve their history.")

        # --------------------------------------------------------------
        # 6. ActionManager partial reset
        # --------------------------------------------------------------
        env.action_manager.reset(
            reset_env_ids #type:ignore
        )

        action_3 = torch.full(
            (env.num_envs, 12),
            0.2,
            device=env.device,
        )

        # Selected envs currently contain zero after reset.
        # Other envs still contain action_2.
        expected_previous_action = action_2.clone()
        expected_previous_action[
            reset_env_ids
        ] = 0.0

        env.action_manager.process_action(
            action_3
        )

        action_rate_3 = evaluate_term(
            "action_rate"
        )

        expected_action_rate_3 = (
            torch.linalg.vector_norm(
                action_3
                - expected_previous_action,
                dim=-1,
            )
        )

        torch.testing.assert_close(
            action_rate_3,
            expected_action_rate_3,
            rtol=1.0e-6,
            atol=1.0e-6,
            msg=(
                "ActionManager partial reset did not produce "
                "the expected action-rate history."
            ),
        )

        print("[PASS] ActionManager partial reset is correct.")
        print("[PASS] Joint/action/torque reward contract validated.")

    finally:
        env.close()

def _validate_contact_pose_rewards() -> None:
    """Validate collision, joint-pose and foot-stumble rewards."""

    env_cfg = Go2ExtremeParkourTeacherEnvCfg()

    env_cfg.scene.num_envs = 6
    env_cfg.sim.device = args_cli.device

    terrain_generator_cfg = env_cfg.scene.terrain.terrain_generator
    if terrain_generator_cfg is None:
        raise RuntimeError(
            "Extreme Parkour requires a terrain generator."
        )

    terrain_generator_cfg.num_rows = 2
    terrain_generator_cfg.num_cols = 5
    terrain_generator_cfg.curriculum = True
    env_cfg.scene.terrain.max_init_terrain_level = 0

    env = ManagerBasedRLEnv(cfg=env_cfg)

    try:
        env.reset()

        robot: Articulation = env.scene["robot"]
        contact_sensor = env.scene.sensors[
            "contact_forces"
        ]

        expected_term_names = [
            "tracking_goal_vel",
            "tracking_yaw",
            "lin_vel_z",
            "ang_vel_xy",
            "orientation",
            "dof_acc",
            "collision",
            "action_rate",
            "delta_torques",
            "torques",
            "hip_pos",
            "dof_error",
            "feet_stumble",
            "feet_edge",
        ]

        if env.reward_manager.active_terms != expected_term_names:
            raise RuntimeError(
                "Incorrect reward term order:\n"
                f"expected={expected_term_names}\n"
                f"actual={env.reward_manager.active_terms}"
            )

        expected_weights = {
            "collision": -10.0,
            "hip_pos": -0.5,
            "dof_error": -0.04,
            "feet_stumble": -1.0,
        }

        for term_name, expected_weight in (
            expected_weights.items()
        ):
            actual_weight = (
                env.reward_manager
                .get_term_cfg(term_name)
                .weight
            )

            if not math.isclose(
                actual_weight,
                expected_weight,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(
                    f"Reward '{term_name}' expected weight "
                    f"{expected_weight}, got {actual_weight}."
                )

        print("[PASS] Fourteen reward terms are registered.")
        print("[PASS] Contact/pose reward weights are correct.")

        def evaluate_term(
            term_name: str,
        ) -> torch.Tensor:
            term_cfg = env.reward_manager.get_term_cfg(
                term_name
            )

            result = term_cfg.func(
                env,
                **term_cfg.params,
            )

            if result.shape != (env.num_envs,):
                raise RuntimeError(
                    f"Reward '{term_name}' returned shape "
                    f"{tuple(result.shape)}."
                )

            if not torch.isfinite(result).all():
                raise RuntimeError(
                    f"Reward '{term_name}' contains NaN or Inf."
                )

            return result

        # --------------------------------------------------------------
        # 1. Collision body selection
        # --------------------------------------------------------------
        collision_cfg = (
            env.reward_manager
            .get_term_cfg("collision")
        )

        collision_sensor_cfg = (
            collision_cfg.params["sensor_cfg"]
        )

        if isinstance(
            collision_sensor_cfg.body_ids,
            slice,
        ):
            raise RuntimeError(
                "collision must resolve explicit body IDs."
            )

        collision_body_ids = list(
            collision_sensor_cfg.body_ids
        )

        collision_body_names = tuple(
            contact_sensor.body_names[index] # type:ignore
            for index in collision_body_ids
        )

        expected_collision_bodies = {
            "base",
            "FL_thigh",
            "FR_thigh",
            "RL_thigh",
            "RR_thigh",
            "FL_calf",
            "FR_calf",
            "RL_calf",
            "RR_calf",
        }

        if set(collision_body_names) != (
            expected_collision_bodies
        ):
            raise RuntimeError(
                "Incorrect collision body selection:\n"
                f"expected={sorted(expected_collision_bodies)}\n"
                f"actual={sorted(collision_body_names)}"
            )

        if len(collision_body_ids) != 9:
            raise RuntimeError(
                "Expected nine collision bodies, "
                f"got {len(collision_body_ids)}."
            )

        if any(
            name.endswith("_foot")
            for name in collision_body_names
        ):
            raise RuntimeError(
                "Feet must not be included in collision bodies."
            )

        print("[PASS] Collision resolves base + 4 thighs + 4 calves.")
        print("[PASS] Normal foot contacts are excluded.")

        # --------------------------------------------------------------
        # 2. Controlled collision counts
        # --------------------------------------------------------------
        current_forces = (
            contact_sensor.data.net_forces_w
        )

        current_forces.zero_()

        expected_collision_counts = torch.tensor(
            [0.0, 1.0, 2.0, 3.0, 8.0, 9.0],
            device=env.device,
        )

        for env_index, collision_count in enumerate(
            expected_collision_counts.tolist()
        ):
            count = int(collision_count)

            if count > 0:
                current_forces[
                    env_index,
                    collision_body_ids[:count],
                    0,
                ] = 0.2

        actual_collision = evaluate_term(
            "collision"
        )

        torch.testing.assert_close(
            actual_collision,
            expected_collision_counts,
            rtol=0.0,
            atol=0.0,
            msg=(
                "collision must count each penalized body "
                "whose force magnitude exceeds 0.1."
            ),
        )

        # Strict threshold test: exactly 0.1 must not trigger.
        current_forces.zero_()

        current_forces[
            0,
            collision_body_ids[0],
            0,
        ] = 0.1

        current_forces[
            1,
            collision_body_ids[0],
            0,
        ] = 0.1001

        threshold_collision = evaluate_term(
            "collision"
        )

        if threshold_collision[0] != 0.0:
            raise RuntimeError(
                "Force exactly equal to threshold 0.1 "
                "must not trigger collision."
            )

        if threshold_collision[1] != 1.0:
            raise RuntimeError(
                "Force above threshold 0.1 "
                "must trigger collision."
            )

        print("[PASS] Collision counts individual bodies.")
        print("[PASS] Collision threshold uses strict > 0.1.")

        # --------------------------------------------------------------
        # 3. Hip and full-joint selections
        # --------------------------------------------------------------
        hip_cfg = env.reward_manager.get_term_cfg(
            "hip_pos"
        )
        dof_cfg = env.reward_manager.get_term_cfg(
            "dof_error"
        )

        hip_asset_cfg = hip_cfg.params["asset_cfg"]
        dof_asset_cfg = dof_cfg.params["asset_cfg"]

        if isinstance(hip_asset_cfg.joint_ids, slice):
            raise RuntimeError(
                "hip_pos must resolve four explicit joints."
            )

        if isinstance(dof_asset_cfg.joint_ids, slice):
            raise RuntimeError(
                "dof_error must resolve 12 explicit joints."
            )

        hip_joint_ids = list(
            hip_asset_cfg.joint_ids
        )
        policy_joint_ids = list(
            dof_asset_cfg.joint_ids
        )

        hip_joint_names = tuple(
            robot.joint_names[index]
            for index in hip_joint_ids
        )

        expected_hip_names = (
            "FR_hip_joint",
            "FL_hip_joint",
            "RR_hip_joint",
            "RL_hip_joint",
        )

        if hip_joint_names != expected_hip_names:
            raise RuntimeError(
                "Incorrect hip joint order:\n"
                f"expected={expected_hip_names}\n"
                f"actual={hip_joint_names}"
            )

        if len(policy_joint_ids) != 12:
            raise RuntimeError(
                f"dof_error expected 12 joints, "
                f"got {len(policy_joint_ids)}."
            )

        print("[PASS] hip_pos selects four hips in policy order.")
        print("[PASS] dof_error selects all 12 policy joints.")

        # --------------------------------------------------------------
        # 4. Controlled joint-position offsets
        # --------------------------------------------------------------
        joint_offset = torch.zeros(
            env.num_envs,
            12,
            device=env.device,
        )

        # Policy layout:
        # 0 FR hip, 1 FR thigh, 2 FR calf,
        # 3 FL hip, 4 FL thigh, 5 FL calf,
        # 6 RR hip, 7 RR thigh, 8 RR calf,
        # 9 RL hip, 10 RL thigh, 11 RL calf.
        hip_policy_indices = [0, 3, 6, 9]

        non_hip_policy_indices = [
            1, 2, 4, 5, 7, 8, 10, 11
        ]

        # env 0: no deviation.
        # env 1: hips only.
        joint_offset[
            1,
            hip_policy_indices,
        ] = 0.1

        # env 2: thigh/calf only.
        joint_offset[
            2,
            non_hip_policy_indices,
        ] = 0.2

        # env 3: both groups with different magnitudes.
        joint_offset[
            3,
            hip_policy_indices,
        ] = 0.3
        joint_offset[
            3,
            non_hip_policy_indices,
        ] = 0.1

        # env 4: unique values catch ordering mistakes.
        joint_offset[4] = torch.linspace(
            -0.3,
            0.3,
            12,
            device=env.device,
        )

        # env 5: every joint has the same deviation.
        joint_offset[5] = -0.2

        controlled_joint_pos = (
            robot.data.default_joint_pos[
                :,
                policy_joint_ids,
            ]
            + joint_offset
        )

        robot.write_joint_position_to_sim(
            controlled_joint_pos,
            joint_ids=policy_joint_ids,
        )

        actual_hip_pos = evaluate_term(
            "hip_pos"
        )

        actual_dof_error = evaluate_term(
            "dof_error"
        )

        expected_hip_pos = torch.sum(
            torch.square(
                joint_offset[
                    :,
                    hip_policy_indices,
                ]
            ),
            dim=-1,
        )

        expected_dof_error = torch.sum(
            torch.square(joint_offset),
            dim=-1,
        )

        torch.testing.assert_close(
            actual_hip_pos,
            expected_hip_pos,
            rtol=1.0e-6,
            atol=1.0e-6,
            msg=(
                "hip_pos does not equal the squared deviation "
                "of the four hip joints."
            ),
        )

        torch.testing.assert_close(
            actual_dof_error,
            expected_dof_error,
            rtol=1.0e-6,
            atol=1.0e-6,
            msg=(
                "dof_error does not equal the squared deviation "
                "of all 12 policy joints."
            ),
        )

        torch.testing.assert_close(
            actual_hip_pos[1],
            torch.tensor(
                0.04,
                device=env.device,
            ),
            rtol=1.0e-6,
            atol=1.0e-6,
        )

        torch.testing.assert_close(
            actual_dof_error[2],
            torch.tensor(
                0.32,
                device=env.device,
            ),
            rtol=1.0e-6,
            atol=1.0e-6,
        )

        print("[PASS] hip_pos uses only four hip joints.")
        print("[PASS] dof_error uses all 12 policy joints.")
        print("[PASS] Joint errors are squared, not absolute.")

        # --------------------------------------------------------------
        # 5. Controlled stumble force directions
        # --------------------------------------------------------------
        stumble_cfg = (
            env.reward_manager
            .get_term_cfg("feet_stumble")
        )

        stumble_sensor_cfg = (
            stumble_cfg.params["sensor_cfg"]
        )

        if isinstance(
            stumble_sensor_cfg.body_ids,
            slice,
        ):
            raise RuntimeError(
                "feet_stumble must resolve four explicit feet."
            )

        foot_body_ids = list(
            stumble_sensor_cfg.body_ids
        )

        resolved_foot_names = tuple(
            contact_sensor.body_names[index] # type:ignore
            for index in foot_body_ids
        )

        if resolved_foot_names != (
            EXTREME_PARKOUR_POLICY_FOOT_NAMES
        ):
            raise RuntimeError(
                "Incorrect foot order:\n"
                f"expected={EXTREME_PARKOUR_POLICY_FOOT_NAMES}\n"
                f"actual={resolved_foot_names}"
            )

        current_forces.zero_()

        # env 0: no contact.
        #
        # env 1: normal vertical ground contact:
        #         horizontal=0, vertical=100 -> no stumble.
        current_forces[
            1,
            foot_body_ids[0],
        ] = torch.tensor(
            [0.0, 0.0, 100.0],
            device=env.device,
        )

        # env 2: 401 > 4 * 100 -> stumble.
        current_forces[
            2,
            foot_body_ids[1],
        ] = torch.tensor(
            [401.0, 0.0, 100.0],
            device=env.device,
        )

        # env 3: exactly 400 == 4 * 100 -> no stumble.
        current_forces[
            3,
            foot_body_ids[2],
        ] = torch.tensor(
            [400.0, 0.0, 100.0],
            device=env.device,
        )

        # env 4: sqrt(300^2 + 400^2) = 500 > 400.
        current_forces[
            4,
            foot_body_ids[3],
        ] = torch.tensor(
            [300.0, 400.0, 100.0],
            device=env.device,
        )

        # env 5: vertical force uses abs(z).
        current_forces[
            5,
            foot_body_ids[0],
        ] = torch.tensor(
            [0.0, 401.0, -100.0],
            device=env.device,
        )

        actual_stumble = evaluate_term(
            "feet_stumble"
        )

        expected_stumble = torch.tensor(
            [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            device=env.device,
        )

        torch.testing.assert_close(
            actual_stumble,
            expected_stumble,
            rtol=0.0,
            atol=0.0,
            msg=(
                "feet_stumble does not implement "
                "norm(force_xy) > 4 * abs(force_z)."
            ),
        )

        print("[PASS] Vertical foot force does not trigger stumble.")
        print("[PASS] Large horizontal foot force triggers stumble.")
        print("[PASS] Boundary equality does not trigger stumble.")
        print("[PASS] feet_stumble uses abs(vertical force).")
        print("[PASS] Contact and joint-pose reward contract validated.")

    finally:
        env.close()

def _validate_feet_edge_reward() -> None:
    """Validate the stateful Extreme Parkour foot-edge penalty."""

    env_cfg = Go2ExtremeParkourTeacherEnvCfg()

    env_cfg.scene.num_envs = 6
    env_cfg.sim.device = args_cli.device

    terrain_generator_cfg = (
        env_cfg.scene.terrain.terrain_generator
    )
    if terrain_generator_cfg is None:
        raise RuntimeError(
            "Extreme Parkour requires a terrain generator."
        )

    # 只创建一个很小的课程地形，减少测试启动成本。
    terrain_generator_cfg.num_rows = 2
    terrain_generator_cfg.num_cols = 5
    terrain_generator_cfg.curriculum = True
    env_cfg.scene.terrain.max_init_terrain_level = 0

    env = ManagerBasedRLEnv(cfg=env_cfg)

    try:
        env.reset()

        feet_edge_cfg = (
            env.reward_manager.get_term_cfg("feet_edge")
        )

        if not math.isclose(
            feet_edge_cfg.weight,
            -1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                "feet_edge weight must be -1.0, "
                f"got {feet_edge_cfg.weight}."
            )

        # 对于 ManagerTermBase 类奖励，RewardManager 初始化时
        # 会把 cfg.func 从类替换成已经创建好的对象。
        feet_edge_term = feet_edge_cfg.func

        if not hasattr(feet_edge_term, "reset"):
            raise RuntimeError(
                "feet_edge must be a stateful ManagerTermBase."
            )

        contact_sensor = env.scene.sensors[
            "contact_forces"
        ]

        sensor_cfg = feet_edge_cfg.params["sensor_cfg"]

        if isinstance(sensor_cfg.body_ids, slice):
            raise RuntimeError(
                "feet_edge must resolve four explicit foot IDs."
            )

        foot_body_ids = list(sensor_cfg.body_ids)

        if len(foot_body_ids) != 4:
            raise RuntimeError(
                "feet_edge expected four feet, "
                f"got {len(foot_body_ids)}."
            )

        resolved_foot_names = tuple(
            contact_sensor.body_names[body_id]  # type: ignore
            for body_id in foot_body_ids
        )

        if resolved_foot_names != (
            EXTREME_PARKOUR_POLICY_FOOT_NAMES
        ):
            raise RuntimeError(
                "Incorrect feet_edge foot order:\n"
                f"expected={EXTREME_PARKOUR_POLICY_FOOT_NAMES}\n"
                f"actual={resolved_foot_names}"
            )

        expected_scanner_names = [
            "foot_height_scanner_FR",
            "foot_height_scanner_FL",
            "foot_height_scanner_RR",
            "foot_height_scanner_RL",
        ]

        scanner_names = feet_edge_cfg.params[
            "scanner_names"
        ]

        if scanner_names != expected_scanner_names:
            raise RuntimeError(
                "Incorrect foot scanner order:\n"
                f"expected={expected_scanner_names}\n"
                f"actual={scanner_names}"
            )

        scanners = [
            env.scene.sensors[scanner_name]
            for scanner_name in scanner_names
        ]

        # ordering="yx" 在当前 GridPatternCfg 实现中产生：
        #
        # x=-0.05: y=-0.05, 0.00, 0.05
        # x= 0.00: y=-0.05, 0.00, 0.05
        # x= 0.05: y=-0.05, 0.00, 0.05
        #
        # 因而 reshape(3, 3) 后，第一个维度就是 x。
        expected_ray_starts = torch.tensor(
            [
                [-0.05, -0.05, 0.0],
                [-0.05,  0.00, 0.0],
                [-0.05,  0.05, 0.0],
                [ 0.00, -0.05, 0.0],
                [ 0.00,  0.00, 0.0],
                [ 0.00,  0.05, 0.0],
                [ 0.05, -0.05, 0.0],
                [ 0.05,  0.00, 0.0],
                [ 0.05,  0.05, 0.0],
            ],
            device=env.device,
        )

        for scanner_name, scanner in zip(
            scanner_names, #type:ignore
            scanners,
        ):
            if scanner.num_rays != 9:
                raise RuntimeError(
                    f"{scanner_name} expected 9 rays, "
                    f"got {scanner.num_rays}."
                )

            torch.testing.assert_close(
                scanner.ray_starts[0],
                expected_ray_starts,
                rtol=0.0,
                atol=1.0e-6,
                msg=(
                    f"{scanner_name} does not use the expected "
                    "3 x 3 local ray ordering."
                ),
            )

        print("[PASS] feet_edge resolves FR/FL/RR/RL feet.")
        print("[PASS] Four scanners use the expected 3 x 3 grid.")

        current_forces = (
            contact_sensor.data.net_forces_w
        )

        terrain_levels = (
            env.scene.terrain.terrain_levels #type:ignore
        )

        # 缓存四个 [num_envs, 9] 高度张量。
        # 测试期间不调用 env.step()，所以不会被传感器刷新覆盖。
        hit_heights = [
            scanner.data.ray_hits_w[..., 2]
            for scanner in scanners
        ]

        def clear_controlled_state() -> None:
            """Restore a flat, contact-free controlled state."""

            current_forces.zero_()
            terrain_levels.fill_(4)

            for hit_height in hit_heights:
                hit_height.zero_()

            feet_edge_term.reset()

        def set_contact(
            env_index: int,
            foot_index: int,
        ) -> None:
            """Set one foot's contact force above the 2 N threshold."""

            current_forces[
                env_index,
                foot_body_ids[foot_index],
                2,
            ] = 10.0

        def set_x_edge(
            env_index: int,
            foot_index: int,
            height_delta: float = 0.1,
        ) -> None:
            """Create a height jump between adjacent x rows."""

            height_grid = hit_heights[
                foot_index
            ][env_index].view(3, 3)

            height_grid.zero_()
            height_grid[2, :] = -height_delta

        def set_y_edge(
            env_index: int,
            foot_index: int,
            height_delta: float = 0.1,
        ) -> None:
            """Create only a lateral y-direction height jump."""

            height_grid = hit_heights[
                foot_index
            ][env_index].view(3, 3)

            height_grid.zero_()
            height_grid[:, 2] = -height_delta

        def evaluate_term() -> torch.Tensor:
            """Evaluate the raw, unweighted feet_edge value."""

            result = feet_edge_term(
                env,
                **feet_edge_cfg.params,
            )

            if result.shape != (env.num_envs,):
                raise RuntimeError(
                    "feet_edge returned shape "
                    f"{tuple(result.shape)}."
                )

            if not torch.isfinite(result).all():
                raise RuntimeError(
                    "feet_edge contains NaN or Inf."
                )

            return result

        # ----------------------------------------------------------
        # 1. 基础判据组合测试
        # ----------------------------------------------------------
        clear_controlled_state()

        # env 0：有接触，但地面完全平坦。
        set_contact(0, 0)

        # env 1：高难度地形、有接触、有 x 方向边缘。
        set_x_edge(1, 0)
        set_contact(1, 0)

        # env 2：边缘和接触都有，但地形等级只有 3。
        set_x_edge(2, 0)
        set_contact(2, 0)
        terrain_levels[2] = 3

        # env 3：有边缘，但没有足端接触。
        set_x_edge(3, 0)

        # env 4：只有 y 方向高度变化。
        # 官方 x_edge_mask 不应触发这一项。
        set_y_edge(4, 0)
        set_contact(4, 0)

        # env 5：两只脚同时接触边缘，返回值应为 2。
        set_x_edge(5, 0)
        set_contact(5, 0)

        set_x_edge(5, 3)
        set_contact(5, 3)

        actual = evaluate_term()

        expected = torch.tensor(
            [0.0, 1.0, 0.0, 0.0, 0.0, 2.0],
            device=env.device,
        )

        torch.testing.assert_close(
            actual,
            expected,
            rtol=0.0,
            atol=0.0,
            msg=(
                "feet_edge basic contact/edge/level "
                "conditions are incorrect."
            ),
        )

        print("[PASS] Flat ground does not trigger feet_edge.")
        print("[PASS] x-direction terrain edges are detected.")
        print("[PASS] y-only changes do not trigger x_edge_mask.")
        print("[PASS] Terrain levels below 4 are gated out.")
        print("[PASS] Multiple contacting feet are counted.")

        # ----------------------------------------------------------
        # 2. 严格高度阈值测试
        # ----------------------------------------------------------
        clear_controlled_state()

        set_x_edge(0, 0, height_delta=0.075)
        set_contact(0, 0)

        set_x_edge(1, 0, height_delta=0.0751)
        set_contact(1, 0)

        threshold_result = evaluate_term()

        if threshold_result[0] != 0.0:
            raise RuntimeError(
                "A height jump exactly equal to 0.075 m "
                "must not trigger the strict > threshold."
            )

        if threshold_result[1] != 1.0:
            raise RuntimeError(
                "A height jump above 0.075 m "
                "must trigger feet_edge."
            )

        print("[PASS] Edge height threshold uses strict > 0.075 m.")

        # ----------------------------------------------------------
        # 3. 有效/无效射线切换与中心射线保护
        # ----------------------------------------------------------
        clear_controlled_state()

        # env 0：中心射线有效，但前方一整排射线未命中。
        missing_grid = hit_heights[
            0
        ][0].view(3, 3)

        missing_grid[2, :] = torch.inf
        set_contact(0, 0)

        # env 1：中心射线本身无效。
        # 即使产生 valid/invalid 切换，也必须被中心射线保护挡掉。
        center_invalid_grid = hit_heights[
            0
        ][1].view(3, 3)

        center_invalid_grid[1, 1] = torch.inf
        set_contact(1, 0)

        missing_result = evaluate_term()

        if missing_result[0] != 1.0:
            raise RuntimeError(
                "A valid/invalid x-ray transition with a valid "
                "center hit must trigger feet_edge."
            )

        if missing_result[1] != 0.0:
            raise RuntimeError(
                "An invalid center ray must suppress feet_edge."
            )

        print("[PASS] Missing neighboring hits detect a cliff edge.")
        print("[PASS] Invalid center hit suppresses false positives.")

        # ----------------------------------------------------------
        # 4. 一步接触迟滞测试
        # ----------------------------------------------------------
        clear_controlled_state()

        set_x_edge(0, 0)
        set_contact(0, 0)

        first_result = evaluate_term()

        # 当前接触消失，但上一策略步仍有接触。
        current_forces.zero_()
        second_result = evaluate_term()

        # 再过一个策略步，当前和上一帧都没有接触。
        third_result = evaluate_term()

        if first_result[0] != 1.0:
            raise RuntimeError(
                "Current contact at an edge must trigger feet_edge."
            )

        if second_result[0] != 1.0:
            raise RuntimeError(
                "feet_edge must retain contact for one policy step."
            )

        if third_result[0] != 0.0:
            raise RuntimeError(
                "feet_edge contact hysteresis lasted "
                "longer than one policy step."
            )

        print("[PASS] Contact hysteresis lasts exactly one call.")

        # ----------------------------------------------------------
        # 5. 部分环境 reset 测试
        # ----------------------------------------------------------
        clear_controlled_state()

        set_x_edge(0, 0)
        set_contact(0, 0)

        set_x_edge(1, 0)
        set_contact(1, 0)

        initial_result = evaluate_term()

        if initial_result[0] != 1.0 or initial_result[1] != 1.0:
            raise RuntimeError(
                "Partial-reset setup did not establish contacts."
            )

        current_forces.zero_()

        reset_env_ids = torch.tensor(
            [0],
            dtype=torch.long,
            device=env.device,
        )

        feet_edge_term.reset(reset_env_ids)

        reset_result = evaluate_term()

        if reset_result[0] != 0.0:
            raise RuntimeError(
                "Reset environment retained stale contact history."
            )

        if reset_result[1] != 1.0:
            raise RuntimeError(
                "Resetting env 0 incorrectly cleared env 1 history."
            )

        print("[PASS] Partial reset clears only selected environments.")
        print("[PASS] feet_edge reward contract validated.")

    finally:
        env.close()

def _validate_positive_reward_clipping() -> None:
    """Validate total clipping without altering raw reward-term logs."""

    # --------------------------------------------------------------
    # 1. Gym registration contract
    # --------------------------------------------------------------
    task_id = "Go2-ExtremeParkour-Teacher-v0"

    task_spec = gym.spec(task_id)

    expected_entry_point = (
        "blank_rl_lab.tasks.manager_based.locomotion."
        "legged.velocity.go2_extreme_parkour_env_cfg:"
        "ExtremeParkourManagerBasedRLEnv"
    )

    if task_spec.entry_point != expected_entry_point:
        raise RuntimeError(
            "Extreme Parkour Gym entry point is incorrect:\n"
            f"expected={expected_entry_point}\n"
            f"actual={task_spec.entry_point}"
        )

    print("[PASS] Gym task uses the task-specific environment class.")

    # --------------------------------------------------------------
    # 2. Create a small controlled environment
    # --------------------------------------------------------------
    env_cfg = Go2ExtremeParkourTeacherEnvCfg()

    env_cfg.scene.num_envs = 6
    env_cfg.sim.device = args_cli.device
    env_cfg.only_positive_rewards = True

    terrain_generator_cfg = (
        env_cfg.scene.terrain.terrain_generator
    )

    if terrain_generator_cfg is None:
        raise RuntimeError(
            "Extreme Parkour requires a terrain generator."
        )

    terrain_generator_cfg.num_rows = 2
    terrain_generator_cfg.num_cols = 5
    terrain_generator_cfg.curriculum = True
    env_cfg.scene.terrain.max_init_terrain_level = 0

    # This test must not reset environments between controlled steps.
    env_cfg.terminations.time_out = None # type:ignore
    env_cfg.terminations.route_complete = None# type:ignore

    env = ExtremeParkourManagerBasedRLEnv(
        cfg=env_cfg,
    )

    try:
        env.reset()

        if (
            env.__class__
            is not ExtremeParkourManagerBasedRLEnv
        ):
            raise RuntimeError(
                "The test did not create the custom environment."
            )

        if not hasattr(env, "raw_reward_buf"):
            raise RuntimeError(
                "Custom environment has no raw_reward_buf."
            )

        if env.raw_reward_buf.shape != (
            env.num_envs,
        ):
            raise RuntimeError(
                "raw_reward_buf expected shape "
                f"({env.num_envs},), got "
                f"{tuple(env.raw_reward_buf.shape)}."
            )

        # ----------------------------------------------------------
        # 3. Replace reward values with deterministic test values
        # ----------------------------------------------------------
        #
        # RewardManager normally calculates:
        #
        # value = reward_function(env) * weight * dt
        #
        # To make the final per-step values exactly equal to
        # controlled_raw_reward, the test reward function returns:
        #
        # controlled_raw_reward / dt
        #
        controlled_raw_reward = torch.tensor(
            [
                -0.20,
                -0.01,
                 0.00,
                 0.01,
                 0.20,
                 0.60,
            ],
            dtype=torch.float,
            device=env.device,
        )

        if controlled_raw_reward.shape != (
            env.num_envs,
        ):
            raise RuntimeError(
                "Controlled reward vector does not match "
                "the number of environments."
            )

        # Set every existing reward weight to zero. RewardManager
        # will then skip all physical reward functions.
        for term_name in (
            env.reward_manager.active_terms
        ):
            term_cfg = (
                env.reward_manager.get_term_cfg(
                    term_name
                )
            )
            term_cfg.weight = 0.0

        controlled_term_name = "tracking_goal_vel"

        controlled_term_cfg = (
            env.reward_manager.get_term_cfg(
                controlled_term_name
            )
        )

        def controlled_reward_term(
            controlled_env,
        ) -> torch.Tensor:
            return (
                controlled_raw_reward
                / controlled_env.step_dt
            )

        controlled_term_cfg.func = (
            controlled_reward_term
        )
        controlled_term_cfg.params = {}
        controlled_term_cfg.weight = 1.0

        zero_action = torch.zeros(
            env.num_envs,
            EXTREME_PARKOUR_NUM_ACTIONS,
            dtype=torch.float,
            device=env.device,
        )

        # ----------------------------------------------------------
        # 4. Test clipping enabled
        # ----------------------------------------------------------
        (
            observations,
            returned_reward,
            terminated,
            time_outs,
            extras,
        ) = env.step(zero_action)

        del observations, extras

        expected_clipped_reward = (
            controlled_raw_reward.clamp_min(0.0)
        )

        torch.testing.assert_close(
            env.raw_reward_buf,
            controlled_raw_reward,
            rtol=0.0,
            atol=1.0e-7,
            msg=(
                "raw_reward_buf does not preserve the "
                "pre-clipping total reward."
            ),
        )

        torch.testing.assert_close(
            returned_reward,
            expected_clipped_reward,
            rtol=0.0,
            atol=1.0e-7,
            msg=(
                "The reward returned to PPO is not "
                "clamp_min(raw_reward, 0)."
            ),
        )

        torch.testing.assert_close(
            env.reward_buf,
            expected_clipped_reward,
            rtol=0.0,
            atol=1.0e-7,
            msg=(
                "env.reward_buf does not contain the "
                "clipped reward."
            ),
        )

        if torch.any(returned_reward < 0.0):
            raise RuntimeError(
                "Clipped reward still contains "
                "negative values."
            )

        if torch.any(terminated):
            raise RuntimeError(
                "Controlled clipping test unexpectedly terminated."
            )

        if torch.any(time_outs):
            raise RuntimeError(
                "Controlled clipping test unexpectedly timed out."
            )

        print("[PASS] Negative total rewards are clipped to zero.")
        print("[PASS] Zero and positive rewards are unchanged.")
        print("[PASS] raw_reward_buf preserves pre-clipping totals.")

        # ----------------------------------------------------------
        # 5. Check current-step per-term logs
        # ----------------------------------------------------------
        #
        # RewardManager.get_active_iterable_terms() exposes
        # _step_reward. IsaacLab stores:
        #
        # _step_reward = weighted_value / dt
        #
        # Therefore the expected logged value is:
        #
        # controlled_raw_reward / env.step_dt
        #
        logged_term_rates = []

        for env_index in range(env.num_envs):
            iterable_terms = (
                env.reward_manager
                .get_active_iterable_terms(
                    env_index
                )
            )

            term_values = {
                name: values[0]
                for name, values in iterable_terms
            }

            logged_term_rates.append(
                term_values[controlled_term_name]
            )

        actual_logged_term_rates = torch.tensor(
            logged_term_rates,
            dtype=torch.float,
            device=env.device,
        )

        expected_logged_term_rates = (
            controlled_raw_reward
            / env.step_dt
        )

        torch.testing.assert_close(
            actual_logged_term_rates,
            expected_logged_term_rates,
            rtol=0.0,
            atol=1.0e-6,
            msg=(
                "Per-term logs were modified by "
                "total reward clipping."
            ),
        )

        # In particular, negative term logs must remain negative.
        if actual_logged_term_rates[0] >= 0.0:
            raise RuntimeError(
                "Negative raw reward-term log was clipped."
            )

        print("[PASS] Per-term step logs remain unclipped.")

        # ----------------------------------------------------------
        # 6. Check episode logging
        # ----------------------------------------------------------
        #
        # RewardManager.reset() returns:
        #
        # mean(episode_sum[env_ids]) / max_episode_length_s
        #
        all_env_ids = torch.arange(
            env.num_envs,
            dtype=torch.long,
            device=env.device,
        )

        episode_logs = env.reward_manager.reset(
            all_env_ids # type:ignore
        )

        episode_log_key = (
            "Episode_Reward/"
            + controlled_term_name
        )

        if episode_log_key not in episode_logs:
            raise RuntimeError(
                f"Missing episode log '{episode_log_key}'."
            )

        expected_episode_log = (
            torch.mean(controlled_raw_reward)
            / env.max_episode_length_s
        )

        torch.testing.assert_close(
            episode_logs[episode_log_key],
            expected_episode_log,
            rtol=0.0,
            atol=1.0e-7,
            msg=(
                "Episode reward log does not contain "
                "the raw weighted reward sum."
            ),
        )

        print("[PASS] Episode reward logs remain unclipped.")

        # ----------------------------------------------------------
        # 7. Test disabling the configuration switch
        # ----------------------------------------------------------
        env.cfg.only_positive_rewards = False

        (
            _,
            unclipped_returned_reward,
            terminated,
            time_outs,
            _,
        ) = env.step(zero_action)

        torch.testing.assert_close(
            unclipped_returned_reward,
            controlled_raw_reward,
            rtol=0.0,
            atol=1.0e-7,
            msg=(
                "Disabling only_positive_rewards did not "
                "return the raw total reward."
            ),
        )

        torch.testing.assert_close(
            env.raw_reward_buf,
            controlled_raw_reward,
            rtol=0.0,
            atol=1.0e-7,
        )

        if torch.any(terminated) or torch.any(time_outs):
            raise RuntimeError(
                "Unclipped controlled step unexpectedly ended."
            )

        env.cfg.only_positive_rewards = True

        print("[PASS] only_positive_rewards=False disables clipping.")
        print("[PASS] Positive total reward contract validated.")

    finally:
        env.close()

def _validate_push_randomization() -> None:
    """Validate push period, axes and velocity-change range."""

    env_cfg = Go2ExtremeParkourTeacherEnvCfg()
    env_cfg.scene.num_envs = 8
    env_cfg.sim.device = args_cli.device

    terrain_generator_cfg = (
        env_cfg.scene.terrain.terrain_generator
    )
    if terrain_generator_cfg is None:
        raise RuntimeError(
            "Extreme Parkour requires a terrain generator."
        )

    terrain_generator_cfg.num_rows = 2
    terrain_generator_cfg.num_cols = 5
    terrain_generator_cfg.curriculum = True
    env_cfg.scene.terrain.max_init_terrain_level = 0

    env = ManagerBasedRLEnv(cfg=env_cfg)

    try:
        env.reset()

        robot: Articulation = env.scene["robot"]

        push_cfg = env.cfg.events.push_robot # type:ignore

        if push_cfg.mode != "interval":
            raise RuntimeError(
                "Push event is not an interval event."
            )

        if push_cfg.interval_range_s != (8.0, 8.0):
            raise RuntimeError(
                "Push interval is not exactly 8 seconds."
            )

        policy_dt = env.step_dt

        torch.testing.assert_close(
            torch.tensor(policy_dt),
            torch.tensor(0.02),
            rtol=0.0,
            atol=1.0e-9,
        )

        expected_steps = round(8.0 / policy_dt)

        if expected_steps != 400:
            raise RuntimeError(
                f"Expected 400 policy steps, got {expected_steps}."
            )

        zero_velocity = torch.zeros_like(
            robot.data.root_vel_w
        )
        robot.write_root_velocity_to_sim(
            zero_velocity
        )

        initial_velocity = (
            robot.data.root_vel_w.clone()
        )

        # The event must not trigger during the first 399 calls.
        for _ in range(expected_steps - 1):
            env.event_manager.apply(
                mode="interval",
                dt=policy_dt,
            )

        before_trigger = (
            robot.data.root_vel_w.clone()
        )

        torch.testing.assert_close(
            before_trigger,
            initial_velocity,
            rtol=0.0,
            atol=1.0e-7,
            msg="Push happened before 8 seconds.",
        )

        # The 400th 20-ms update reaches 8 seconds.
        env.event_manager.apply(
            mode="interval",
            dt=policy_dt,
        )

        after_trigger = (
            robot.data.root_vel_w.clone()
        )

        delta_velocity = (
            after_trigger - before_trigger
        )

        # x/y are randomized independently.
        if torch.any(
            torch.abs(delta_velocity[:, 0:2])
            > 0.5 + 1.0e-6
        ):
            raise RuntimeError(
                "Planar velocity change exceeds 0.5 m/s "
                "on one axis."
            )

        # z and angular velocity must remain unchanged.
        torch.testing.assert_close(
            delta_velocity[:, 2:6],
            torch.zeros_like(
                delta_velocity[:, 2:6]
            ),
            rtol=0.0,
            atol=1.0e-7,
            msg=(
                "Push unexpectedly changed z or "
                "angular velocity."
            ),
        )

        if not torch.any(
            torch.abs(delta_velocity[:, 0:2])
            > 1.0e-6
        ):
            raise RuntimeError(
                "The interval event did not apply a push."
            )

        if not torch.isfinite(
            delta_velocity
        ).all():
            raise RuntimeError(
                "Push produced NaN or Inf."
            )

        print("[PASS] Push interval is exactly 8 seconds.")
        print("[PASS] 8 seconds equals 400 policy steps.")
        print("[PASS] Only planar velocity is changed.")
        print("[PASS] Per-axis change is within ±0.5 m/s.")

    finally:
        env.close()

def _validate_action_delay() -> None:
    """Validate zero-delay and one-policy-step delay."""

    env_cfg = Go2ExtremeParkourTeacherEnvCfg()
    env_cfg.scene.num_envs = 4
    env_cfg.sim.device = args_cli.device

    terrain_generator_cfg = (
        env_cfg.scene.terrain.terrain_generator
    )
    if terrain_generator_cfg is None:
        raise RuntimeError(
            "Extreme Parkour requires a terrain generator."
        )

    terrain_generator_cfg.num_rows = 2
    terrain_generator_cfg.num_cols = 5
    terrain_generator_cfg.curriculum = True
    env_cfg.scene.terrain.max_init_terrain_level = 0

    env = ManagerBasedRLEnv(cfg=env_cfg)

    try:
        env.reset()

        action_term = (
            env.action_manager.get_term("joint_pos")
        )

        if not isinstance(
            action_term,
            mdp.Go2DelayedJointPositionAction, # type:ignore
        ):
            raise RuntimeError(
                "Extreme Parkour is not using the "
                "delayed joint-position ActionTerm."
            )

        env_ids = torch.arange(
            env.num_envs,
            device=env.device,
        )

        action_a = torch.full(
            (env.num_envs, 12),
            0.4,
            device=env.device,
        )

        action_b = torch.full(
            (env.num_envs, 12),
            -0.3,
            device=env.device,
        )

        # ----------------------------------------------------------
        # No-delay phase
        # ----------------------------------------------------------
        action_term.cfg.randomize_action_delay = False # type:ignore
        action_term.reset(env_ids) # type:ignore

        action_term.process_actions(action_a)
        action_term.apply_actions()

        torch.testing.assert_close(
            action_term._applied_raw_actions, # type:ignore
            action_a,
            rtol=0.0,
            atol=0.0,
            msg="No-delay mode did not apply current action.",
        )

        print("[PASS] First phase applies a_t immediately.")

        # ----------------------------------------------------------
        # Fixed 4-physics-step = 20-ms delay
        # ----------------------------------------------------------
        action_term.cfg.randomize_action_delay = True # type:ignore
        action_term.cfg.max_delay_steps = 4 # type:ignore
        action_term.reset(env_ids) # type:ignore

        # First command after reset: previous action is zero.
        action_term.process_actions(action_a)

        for _ in range(4):
            action_term.apply_actions()

            torch.testing.assert_close(
                action_term._applied_raw_actions, # type:ignore
                torch.zeros_like(action_a),
                rtol=0.0,
                atol=0.0,
                msg=(
                    "Current action leaked through before "
                    "the 20-ms delay elapsed."
                ),
            )

        # During the next policy period, action_a must be used
        # while action_b is waiting.
        action_term.process_actions(action_b)

        for _ in range(4):
            action_term.apply_actions()

            torch.testing.assert_close(
                action_term._applied_raw_actions, # type:ignore
                action_a,
                rtol=0.0,
                atol=0.0,
                msg=(
                    "Fixed action delay did not apply "
                    "the previous policy action."
                ),
            )

        expected_delay = (
            4 * EXTREME_PARKOUR_SIM_DT
        )

        torch.testing.assert_close(
            torch.tensor(expected_delay),
            torch.tensor(
                EXTREME_PARKOUR_POLICY_DT
            ),
            rtol=0.0,
            atol=1.0e-9,
        )

        print("[PASS] Four physics steps equal 20 ms.")
        print("[PASS] Delayed mode applies a_(t-1).")
        print("[PASS] Action-delay reset clears stale actions.")

    finally:
        env.close()
def main() -> None:
    if args_cli.check_action_delay:
        with torch.inference_mode():
            _validate_action_delay()
        return

    if args_cli.check_push_randomization:
        with torch.inference_mode():
            _validate_push_randomization()
        return

    if args_cli.check_positive_reward_clipping:
        with torch.inference_mode():
            _validate_positive_reward_clipping()
        return

    if args_cli.check_feet_edge_reward:
        with torch.inference_mode():
            _validate_feet_edge_reward()
        return

    if args_cli.check_contact_pose_rewards:
        with torch.inference_mode():
            _validate_contact_pose_rewards()
        return

    if args_cli.check_joint_regularization_rewards:
        with torch.inference_mode():
            _validate_joint_regularization_rewards()
        return

    if args_cli.check_body_regularization_rewards:
        with torch.inference_mode():
            _validate_body_regularization_rewards()
        return

    if args_cli.check_navigation_rewards:
        with torch.inference_mode():
            _validate_navigation_rewards()
        return

    if args_cli.check_proprio_history_contract:
        with torch.inference_mode():
            _validate_proprio_history_contract()
        return

    if args_cli.check_priv_latent_contract:
        with torch.inference_mode():
            _validate_priv_latent_contract()
        return

    if args_cli.check_priv_explicit_contract:
        with torch.inference_mode():
            _validate_priv_explicit_contract()
        return

    if args_cli.check_terrain_scan_contract:
        with torch.inference_mode():
            _validate_terrain_scan_contract()
        return

    if args_cli.check_proprio_contract:
        with torch.inference_mode():
            _validate_proprio_contract()
        return

    if args_cli.check_waypoint_command:
        with torch.inference_mode():
            _validate_waypoint_command()
        return

    if args_cli.zero_policy_steps < 0 or args_cli.random_policy_steps < 0:
        raise ValueError("Policy-step counts must be non-negative.")
    if not 0.0 <= args_cli.random_action_amplitude <= 1.0:
        raise ValueError("--random_action_amplitude must be in [0, 1].")

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=EXTREME_PARKOUR_SIM_DT, device=args_cli.device)
    )
    sim.set_camera_view(eye=(2.5, 2.5, 1.8), target=(0.0, 0.0, 0.35))

    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
    )
    ground_cfg.func("/World/Ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = EXTREME_PARKOUR_GO2_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)
    sim.reset()

    _print_names("Runtime body names", robot.body_names)
    _print_names("Runtime joint names", robot.joint_names)

    if len(robot.joint_names) != EXTREME_PARKOUR_NUM_ACTIONS:
        raise RuntimeError(
            f"Expected {EXTREME_PARKOUR_NUM_ACTIONS} joints, found {len(robot.joint_names)}: {robot.joint_names}"
        )
    foot_indices, foot_names = robot.find_bodies(".*_foot", preserve_order=True)
    if len(foot_names) != 4 or set(foot_names) != set(EXPECTED_FOOT_NAMES):
        raise RuntimeError(f"Expected feet {EXPECTED_FOOT_NAMES}, found {tuple(foot_names)}")
    base_indices, base_names = robot.find_bodies("base", preserve_order=True)
    if len(base_names) != 1:
        raise RuntimeError(f"Expected one body named 'base', found {base_names}")

    actuator_cfg = robot_cfg.actuators["legs"]
    shared_actuator_cfg = UNITREE_GO2_CFG.actuators["legs"]
    if shared_actuator_cfg.stiffness != 25.0 or shared_actuator_cfg.damping != 0.5:
        raise RuntimeError(
            "Shared UNITREE_GO2_CFG was unexpectedly modified: "
            f"stiffness={shared_actuator_cfg.stiffness}, damping={shared_actuator_cfg.damping}"
        )
    print("\nFrozen control contract:")
    print(f"  sim_dt             = {EXTREME_PARKOUR_SIM_DT:.6f} s")
    print(f"  decimation         = {EXTREME_PARKOUR_DECIMATION}")
    print(f"  policy_dt          = {EXTREME_PARKOUR_POLICY_DT:.6f} s")
    print(f"  policy_frequency   = {1.0 / EXTREME_PARKOUR_POLICY_DT:.1f} Hz")
    print(f"  action_scale       = {EXTREME_PARKOUR_ACTION_SCALE:.3f} rad")
    print(f"  num_actions        = {EXTREME_PARKOUR_NUM_ACTIONS}")
    print(f"  stiffness          = {actuator_cfg.stiffness}")
    print(f"  damping            = {actuator_cfg.damping}")
    print(f"  effort_limit       = {actuator_cfg.effort_limit}")
    print(f"  velocity_limit     = {actuator_cfg.velocity_limit}")
    print(f"  base_body_index    = {base_indices[0]}")
    print(f"  foot_body_indices  = {foot_indices}")
    print("  shared_go2_gains   = 25.0 / 0.5 (unchanged)")

    _reset_to_default(robot)
    zero_actions = torch.zeros((1, EXTREME_PARKOUR_NUM_ACTIONS), device=robot.device)
    for _ in range(args_cli.zero_policy_steps):
        _step_policy(robot, sim, zero_actions)
    _assert_finite(robot, "zero-action phase")
    print(f"[PASS] zero-action phase: {args_cli.zero_policy_steps} policy steps")

    _reset_to_default(robot)
    generator = torch.Generator(device=robot.device)
    generator.manual_seed(args_cli.seed)
    for _ in range(args_cli.random_policy_steps):
        actions = 2.0 * torch.rand(
            (1, EXTREME_PARKOUR_NUM_ACTIONS), device=robot.device, generator=generator
        ) - 1.0
        actions *= args_cli.random_action_amplitude
        _step_policy(robot, sim, actions)
    _assert_finite(robot, "random-action phase")
    print(f"[PASS] random-action phase: {args_cli.random_policy_steps} policy steps")
    print("[PASS] Extreme Parkour Go2 stage-one contract validated.")

    if args_cli.keep_alive:
        print("Close Isaac Sim or press Ctrl+C to stop.")
        while simulation_app.is_running():
            sim.step()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
