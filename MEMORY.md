# Project Memory

This file is the durable context for this workspace. After context compaction or a new session, read this file first, then inspect the live code before editing.

## Big Picture

Current goal: port the external Go2 Isaac Gym project into this IsaacLab workspace.

External reference project:

- `/home/robot/workshop/Robot/Reinforcementlearn/go2_rl_gym`

Current IsaacLab workspace:

- `/home/robot/workshop/Robot/IsaacLab/blank_rl_lab/blank_rl_lab`

Important rule: this memory is context, not ground truth. Always re-read the current files before changing code because the workspace has active uncommitted edits.

## External Go2 Project: Key Files

- Main Go2 config: `/home/robot/workshop/Robot/Reinforcementlearn/go2_rl_gym/legged_gym/envs/go2/go2_config.py`
- Go2 observation override and custom rewards: `/home/robot/workshop/Robot/Reinforcementlearn/go2_rl_gym/legged_gym/envs/go2/go2_env.py`
- Base legged robot config: `/home/robot/workshop/Robot/Reinforcementlearn/go2_rl_gym/legged_gym/envs/base/legged_robot_config.py`
- Base legged robot logic: `/home/robot/workshop/Robot/Reinforcementlearn/go2_rl_gym/legged_gym/envs/base/legged_robot.py`
- Terrain generator: `/home/robot/workshop/Robot/Reinforcementlearn/go2_rl_gym/legged_gym/utils/terrain.py`
- Task registration: `/home/robot/workshop/Robot/Reinforcementlearn/go2_rl_gym/legged_gym/envs/__init__.py`
- Mujoco deploy config: `/home/robot/workshop/Robot/Reinforcementlearn/go2_rl_gym/deploy/deploy_mujoco/configs/go2.yaml`
- Real deploy config: `/home/robot/workshop/Robot/Reinforcementlearn/go2_rl_gym/deploy/deploy_real/configs/go2.yaml`

Registered external tasks:

- `go2`: `Go2Robot + GO2Cfg + GO2CfgPPO`
- `go2_cts`: `Go2Robot + GO2Cfg + GO2CfgCTS`
- `go2_moe_cts`, `go2_moe_ng_cts`, `go2_mcp_cts`, `go2_ac_moe_cts`, `go2_dual_moe_cts`: advanced CTS/MoE variants

Recommended migration order: first reproduce the environment contract from `GO2Cfg` and `Go2Robot`, then plain PPO/CTS configs, then advanced MoE/CTS runners if needed.

## Current IsaacLab Files

- Existing rough reference: `source/blank_rl_lab/blank_rl_lab/tasks/manager_based/locomotion/velocity/go2_teacher_student_cfg.py`
- Existing task registry: `source/blank_rl_lab/blank_rl_lab/tasks/manager_based/locomotion/velocity/__init__.py`
- Existing MDP package: `source/blank_rl_lab/blank_rl_lab/tasks/manager_based/locomotion/velocity/mdp/`
- Existing agents: `source/blank_rl_lab/blank_rl_lab/tasks/manager_based/locomotion/velocity/agents/`
- Likely target env cfg to create or maintain: `source/blank_rl_lab/blank_rl_lab/tasks/manager_based/locomotion/velocity/go2_rl_gym_velocity_cfg.py`

Known current local state when this memory was updated:

- `MEMORY.md` is untracked.
- `source/.../velocity/mdp/commands.py` is modified.
- `source/.../velocity/go2_cts_moe_env_cfg.py` is untracked.
- Do not revert or overwrite unrelated user work.

## External Go2 Env Contract

From `GO2Cfg.env`:

- `num_envs = 8192`
- `num_observations = 45`
- `num_privileged_obs = 263`
- `num_actions = 12` inherited from base cfg
- `episode_length_s = 25`

Actor observation in `Go2Robot.compute_observations()` is exactly 45 dims:

- `base_ang_vel * 0.25`: 3 dims
- `projected_gravity`: 3 dims
- `commands[:, :3] * [2.0, 2.0, 0.25]`: 3 dims
- `(dof_pos - default_dof_pos) * 1.0`: 12 dims
- `dof_vel * 0.05`: 12 dims
- `actions`: 12 dims

Important compatibility rule:

- Do not include `base_lin_vel` in actor obs.
- Do not include height scan in actor obs.
- Actor obs must stay 45 dims if we want compatibility with original policies and deploy configs.

Privileged critic observation is 263 dims:

- `base_lin_vel * 2.0`: 3 dims
- actor-style obs components: 45 dims
- foot contact force norms for 4 feet, scaled by `1e-3`: 4 dims
- normalized torques, `torques / torque_limits`: 12 dims
- motor accelerations, `(last_dof_vel - dof_vel) / dt * 1e-4`: 12 dims
- height measurements: 187 dims

Privileged height formula in the external Go2 env:

- `heights = clip(root_z - 0.5 - measured_heights, -1, 1.0) * 2.5`

Noise on actor obs:

- ang vel noise: `0.2 * 1.0 * 0.25`
- projected gravity noise: `0.05`
- commands: no noise
- dof pos noise: `0.01`
- dof vel noise: `1.5 * 0.05`
- previous actions: no noise

## Robot Init, Control, And Deploy Contract

From `GO2Cfg.init_state`:

- base position: `[0.0, 0.0, 0.42]`
- default joint angles:
  - `FL_hip_joint = 0.1`, `FR_hip_joint = -0.1`, `RL_hip_joint = 0.1`, `RR_hip_joint = -0.1`
  - `FL_thigh_joint = 0.8`, `FR_thigh_joint = 0.8`, `RL_thigh_joint = 1.0`, `RR_thigh_joint = 1.0`
  - all calf joints: `-1.5`
- `turn_over = False` in the main config

From `GO2Cfg.control`:

- position control, `control_type = "P"`
- stiffness: `20.0`
- damping: `0.5`
- action scale: `0.25`
- decimation: `4`

From base sim config:

- sim dt: `0.005`
- policy/control dt: `0.005 * 4 = 0.02`, i.e. 50 Hz
- PhysX: TGS solver, 4 position iterations, 0 velocity iterations

Deploy configs confirm:

- `num_obs = 45`
- `num_actions = 12`
- `action_scale = 0.25`
- default angles ordered as `FL`, `FR`, `RL`, `RR`, each hip/thigh/calf
- Mujoco deploy uses command scale `[2.0, 2.0, 0.25]`
- Real deploy uses command scale `[3.0, 2.0, 0.5]`
- Real deploy has `joint2motor_idx = [3,4,5,0,1,2,9,10,11,6,7,8]`

## External Terrain Contract

From `LeggedRobotCfg.terrain` plus Go2 overrides:

- `mesh_type = "trimesh"`
- `horizontal_scale = 0.1`
- `vertical_scale = 0.005`
- `border_size = 25`
- `curriculum = True`
- `measure_heights = True`
- `terrain_length = 8.0`
- `terrain_width = 8.0`
- `terrain_spacing = 0.5`
- `num_rows = 10`
- `num_cols = 20`
- `max_init_terrain_level = 5`
- `slope_threshold = 1.5`
- `move_down_by_accumulated_xy_command = True`
- friction: static `1.0`, dynamic `1.0`, restitution `0.0`

Go2 terrain proportions, ordered as `[wave, slope, rough_slope, stairs_up, stairs_down, obstacles, stepping_stones, gap, flat]`:

- `[0.05, 0.20, 0.05, 0.25, 0.10, 0.20, 0.0, 0.0, 0.15]`

Terrain generation in `terrain.py`:

- difficulty per row: `i / num_rows`
- terrain choice per col: `j / num_cols + 0.001`
- hard mode is enabled:
  - slope: `0.1 + difficulty * 0.52`
  - step height: `0.05 + 0.23 * difficulty`
  - discrete obstacle height: `0.05 + difficulty * 0.25`
  - wave amplitude: `0.1 + 0.2 * difficulty`
- wave terrain also adds random uniform noise `[-0.05, 0.05]`, step `0.005`, downsampled scale `0.2`
- slope uses pyramid sloped terrain, half positive and half negative by choice split
- rough slope = pyramid slope plus random uniform roughness
- stairs use `step_width = 0.31`, platform size `3.0`; stairs up is implemented by negative step height in Isaac Gym terrain utils
- obstacles use 20 rectangles, width range `[1.0, 2.0]`, platform size `3.0`
- stepping stones and gap proportions are zero in current Go2 config, so they can be postponed
- flat is `pit_terrain(depth=0.0, platform_size=4.0)`

IsaacLab current `go2_teacher_student_cfg.py` has a partial terrain approximation:

- `COBBLESTONE_ROAD_CFG` with tile size `(8.0, 8.0)`, rows `10`, cols `20`, horizontal scale `0.1`, vertical scale `0.005`
- currently uses flat, smooth slopes, random rough, stairs, inverted stairs, and discrete obstacles
- it does not exactly reproduce external `wave`, `rough_slope`, per-row hard ranges, terrain spacing `0.5`, or `slope_threshold = 1.5`
- current IsaacLab rough config uses `max_init_terrain_level = 0`; external Go2 uses `5`

Migration recommendation:

- Smoke test with existing IsaacLab terrain and `max_init_terrain_level = 0`.
- Then switch toward external Go2: `max_init_terrain_level = 5`, matching terrain proportions, slope/step/obstacle difficulty ranges, and terrain spacing if IsaacLab supports it cleanly.
- Exact wave terrain may need a custom IsaacLab terrain function or an acceptable approximation.

## Height Scanner Contract

External Go2 measured points:

- `measured_points_x = [-0.8, -0.7, ..., 0.8]`, 17 values
- `measured_points_y = [-0.5, -0.4, ..., 0.5]`, 11 values
- total height points: `17 * 11 = 187`
- points are rotated by yaw only and sampled around the base

IsaacLab equivalent already exists in `go2_teacher_student_cfg.py`:

- `RayCasterCfg`
- `prim_path = "{ENV_REGEX_NS}/Robot/base"`
- `offset = (0.0, 0.0, 20.0)`
- `ray_alignment = "yaw"`
- `GridPatternCfg(resolution=0.1, size=[1.6, 1.0])`
- `mesh_prim_paths = ["/World/ground"]`

This gives the same 187-point grid. Keep it in critic/privileged observations first, not in actor observations.

## External Commands Contract

From `GO2Cfg.commands`:

- `num_commands = 4`, but actor uses only first 3: lin vel x, lin vel y, yaw vel
- `resampling_time = 5.0`
- `heading_command = False`
- initial ranges:
  - `lin_vel_x = [-0.5, 0.5]`
  - `lin_vel_y = [-0.5, 0.5]`
  - `ang_vel_yaw = [-1.0, 1.0]`
  - `heading = [-1.57, 1.57]`
- zero command curriculum:
  - iter 0 to 1500
  - probability from `0.0` to `0.1`
- `limit_ang_vel_at_zero_command_prob = 0.2`
- `limit_vel_prob = 0.2`
- `limit_vel = {"lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1], "ang_vel_yaw": [-1, 0, 1]}`
- `limit_vel_invert_when_continuous = True`
- `dynamic_resample_commands = True`

Command range curriculum:

- at iter `20000`: x `[-1.0, 1.0]`, y `[-1.0, 1.0]`, yaw `[-1.5, 1.5]`
- at iter `50000`: x `[-2.0, 2.0]`, y `[-1.0, 1.0]`, yaw `[-2.0, 2.0]`

Terrain-specific max command ranges:

- wave/slope/rough_slope: x `[-1.5, 1.5]`, y `[-1.0, 1.0]`, yaw `[-1.5, 1.5]`
- stairs_up/stairs_down/obstacles/stepping_stones/gap: x `[-1.0, 1.0]`, y `[-1.0, 1.0]`, yaw `[-1.5, 1.5]`
- flat: x `[-2.0, 2.0]`, y `[-1.0, 1.0]`, yaw `[-2.0, 2.0]`

External dynamic command resampling tries to avoid running out of terrain by imposing a lower velocity bound based on remaining distance, episode time, and accumulated xy commands. This is more complex than a simple uniform velocity command.

## Terrain Curriculum Contract

External terrain level update in `LeggedRobot._update_terrain_curriculum()`:

- measure `distance = max_move_distance` from env origin
- move up if `distance > terrain_length / 2`
- with Go2 override, move down if:
  - `distance < norm(commands_xy_accumulation) * (resampling_time * (1 - zero_command_proba)) * 0.5`
  - and not moving up
- if robot solves max level, send it to a random terrain level
- levels are clipped at minimum 0

Initial terrain assignment:

- levels round-robin over `0..max_init_terrain_level`
- terrain types distributed by env index over `num_cols`
- terrain type id is derived from the column's generated terrain type

This is not the same as a plain distance-to-command curriculum unless adapted.

## Current IsaacLab Command Migration Status

This section records the current local migration state for the Go2 command sampler. Re-check live files before editing.

Primary files touched for this part:

- `source/blank_rl_lab/blank_rl_lab/tasks/manager_based/locomotion/velocity/mdp/commands.py`
- `source/blank_rl_lab/blank_rl_lab/tasks/manager_based/locomotion/velocity/mdp/curriculums.py`
- `source/blank_rl_lab/blank_rl_lab/tasks/manager_based/locomotion/velocity/go2_cts_moe_env_cfg.py`

Implemented command term:

- `GoStyleVelocityCommand` extends IsaacLab `UniformVelocityCommand`.
- `GoStyleLevelVelocityCommandCfg` points `class_type` to `GoStyleVelocityCommand`.
- Current target env config uses `mdp.GoStyleLevelVelocityCommandCfg` for `base_velocity`.
- Initial Go2-style command settings in the target config:
  - `resampling_time_range=(5.0, 5.0)`
  - `heading_command=False`
  - initial ranges: x `(-0.5, 0.5)`, y `(-0.5, 0.5)`, yaw `(-1.0, 1.0)`
  - `limit_vel_prob=0.2`
  - `dynamic_resample_commands=True`
  - zero command curriculum from `0.0` to `0.1` over iter `0..1500`

Implemented command runtime buffers:

- `commands_xy_accumulation`: accumulates final sampled xy commands over the episode. It is used by terrain curriculum to estimate how far the command stream asked the robot to move.
- `max_move_distance`: tracks the maximum xy distance from `env_origins` reached during the episode. It is used for terrain promotion.
- `last_is_limit_vel`: tracks whether the previous resample for an env was a limit-velocity sample, enabling the continuous-limit inversion behavior.
- `zero_command_proba`: current scalar zero-command probability from the zero command curriculum.
- `env_command_ranges`: per-env actual command ranges after global command curriculum and terrain-specific constraints.

Implemented command sampling features:

- Command range curriculum:
  - iter `20000`: x/y `(-1.0, 1.0)`, yaw `(-1.5, 1.5)`
  - iter `50000`: x `(-2.0, 2.0)`, y `(-1.0, 1.0)`, yaw `(-2.0, 2.0)`
  - Uses `env.common_step_counter // curriculum_iteration_length`, with `curriculum_iteration_length=24`.
- Zero command curriculum:
  - Gradually increases zero xy command probability from `0.0` to `0.1`.
  - Only zeroes x/y velocity. Yaw may be set to a per-env yaw boundary with probability `limit_ang_vel_at_zero_command_prob=0.2`.
  - Adjusts IsaacLab `time_left` for zero-command envs using a `next_time` estimate, so zero commands do not consume too much remaining episode time.
- Limit velocity:
  - Probability window is `limit_vel_prob=0.2`.
  - Uses combinations of x `(-1, 1)`, y `(-1, 1)`, yaw `(-1, 0, 1)`.
  - `-1`, `0`, `1` map to per-env current min, zero, and max command values.
  - If an env is selected for limit velocity consecutively, the command is inverted when `limit_vel_invert_when_continuous=True`.
- Dynamic resampling:
  - Computes remaining target distance from `0.625 * terrain_length - norm(commands_xy_accumulation) * resampling_time`.
  - Computes a per-env velocity lower bound from remaining distance and remaining episode time.
  - Samples x/y from disjoint intervals, avoiding low-magnitude velocity commands when the episode is running out of time.
- Terrain-specific command ranges:
  - Builds a column-to-terrain-name map using IsaacLab `TerrainGenerator` proportions.
  - Uses `env.scene.terrain.terrain_types` as terrain column ids.
  - Applies per-env range intersections:
    - flat: x `(-2.0, 2.0)`, y `(-1.0, 1.0)`, yaw `(-2.0, 2.0)`
    - smooth slopes and random rough: x `(-1.5, 1.5)`, y `(-1.0, 1.0)`, yaw `(-1.5, 1.5)`
    - stairs and discrete obstacles: x `(-1.0, 1.0)`, y `(-1.0, 1.0)`, yaw `(-1.5, 1.5)`

Implemented terrain curriculum integration:

- Added `terrain_levels_by_go2_command` in `mdp/curriculums.py`.
- It reads the command term object via `env.command_manager.get_term("base_velocity")`, not just the command tensor.
- Promotion uses `max_move_distance` and current distance as a safety max.
- Demotion uses `commands_xy_accumulation * resampling_time * (1.0 - zero_command_proba) * 0.5`.
- Target env curriculum should point to `mdp.terrain_levels_by_go2_command` instead of the default `terrain_levels_vel`.
- IsaacLab reset order is useful here: curriculum compute reads the previous episode's command buffers before command reset clears them.

Known cleanup / verification notes:

- `commands.py` currently compiles with `python3 -m py_compile`.
- `_resample_uniform_command()` has a minor redundant yaw sample when `heading_command=False`; final behavior is correct, but it can be cleaned later.
- Current Go2 target uses `heading_command=False`; dynamic and uniform heading branches have been guarded, but heading mode is not the active migration target.
- After future terrain changes, re-check the `terrain_command_ranges` names against the actual `COBBLESTONE_ROAD_CFG.sub_terrains` keys.

## External Reward Contract

From `GO2Cfg.rewards`:

- `only_positive_rewards = False`
- `tracking_sigma = 0.25`
- `base_height_target = 0.38`
- `soft_dof_pos_limit = 0.9`
- `max_contact_force = 147.0`
- curriculum rewards:
  - `lin_vel_z`: scale multiplier from `1.0` to `0.0`, iter 0 to 1500
  - `correct_base_height`: scale multiplier from `1.0` to `10.0`, iter 0 to 5000

Reward scales in active Go2 config:

- `tracking_lin_vel = 1.0`
- `tracking_ang_vel = 0.5`
- `lin_vel_z = -2.0`
- `ang_vel_xy = -0.05`
- `dof_acc = -2.5e-7`
- `dof_power = -2e-5`
- `torques = -1e-4`
- `correct_base_height = -1.0`
- `action_rate = -0.01`
- `action_smoothness = -0.01`
- `collision = -1.0`
- `dof_pos_limits = -2.0`
- `feet_regulation = -0.05`
- `hip_to_default = -0.05`

Reward implementation notes:

- reward scales are multiplied by `dt` in `_prepare_reward_function()`
- tracking linear velocity uses exp of x/y squared errors, divided by possibly dynamic sigmas
- tracking yaw uses exp of yaw error squared divided by sigma
- dynamic sigma depends on command magnitude, terrain id, and terrain level
- `correct_base_height` estimates ground height from height scan points near the robot base, then penalizes squared error from `0.38`
- `base_height` reward exists but Go2 active scale uses `correct_base_height`
- `collision` penalizes thigh/calf contacts
- termination is on base contact force over threshold, plus timeout

Dynamic sigma config:

- `min_lin_vel = 0.5`
- `max_lin_vel = 1.5`
- `min_ang_vel = 1.0`
- `max_ang_vel = 2.0`
- max sigma by terrain id `[wave, slope, rough_slope, stairs_up, stairs_down, obstacles, stepping_stones, gap, flat]`:
  - `[5/12, 1/4, 1/4, 1/2, 1/2, 3/4, 1, 1, 1/4]`

## Current IsaacLab Observation Migration Status

This section records the current local migration state for the Go2 observation contract. Re-check live files before editing.

External Go2 observation contract:

- Actor observation is 45 dims.
- Privileged critic observation is 263 dims.
- Actor does not include base linear velocity.
- Actor does not include height scan.
- Critic includes privileged quantities such as base linear velocity, contact force, torque, acceleration, and terrain height scan.

Actor / `policy` group target:

- `base_ang_vel`: 3 dims, scale `0.25`, noise `[-0.2, 0.2]`
- `projected_gravity`: 3 dims, noise `[-0.05, 0.05]`
- `velocity_commands`: 3 dims, `generated_commands("base_velocity")`, scale `(2.0, 2.0, 0.25)`
- `joint_pos_rel`: 12 dims, noise `[-0.01, 0.01]`
- `joint_vel_rel`: 12 dims, scale `0.05`, noise `[-1.5, 1.5]`
- `last_action`: 12 dims
- Total: `3 + 3 + 3 + 12 + 12 + 12 = 45`

Why `velocity_commands` has three scale values:

- IsaacLab `generated_commands()` returns `env.command_manager.get_command("base_velocity")`.
- `CommandManager.get_command()` returns the command term's `.command`.
- `UniformVelocityCommand.command` returns `vel_command_b`, whose shape is `(num_envs, 3)`.
- Index mapping:
  - `command[:, 0] = lin_vel_x`
  - `command[:, 1] = lin_vel_y`
  - `command[:, 2] = ang_vel_z`
- Therefore scale `(2.0, 2.0, 0.25)` means x/y linear velocity use `obs_scales.lin_vel = 2.0`, yaw angular velocity uses `obs_scales.ang_vel = 0.25`.
- Even if `heading_command=True`, IsaacLab still exposes a 3D velocity command; `heading_target` is an internal helper target.

Critic group target:

- `base_lin_vel`: 3 dims, scale `2.0`
- Actor-style terms: 45 dims, but without observation corruption/noise
- `foot_contact_forces`: 4 dims, norm of foot contact forces times `1e-3`
- `torques`: 12 dims, `applied_torque / joint_effort_limits`
- `joint_acc`: 12 dims, legacy sign/scale `-asset.data.joint_acc * 1e-4`
- `height_scan`: 187 dims, Go2-style `base_z - 0.5 - hit_z`, clipped `[-1.0, 1.0]`, scale `2.5`
- Total: `3 + 45 + 4 + 12 + 12 + 187 = 263`

Recommended local observation helpers in `mdp/observations.py`:

- `go2_height_scan`
- `go2_foot_contact_forces`
- `go2_joint_torques_normalized`
- `go2_joint_acc_legacy_scaled`

Important observation caveats:

- IsaacLab built-in `height_scan()` computes from sensor origin height. Since this env's `height_scanner` has offset `(0.0, 0.0, 20.0)`, it may not match external Go2 height observations. Prefer the Go2 helper using robot base z and ray hit z.
- `go2_joint_acc_legacy_scaled` should use a negative sign because external Go2 used `(last_dof_vel - dof_vel) / dt`, while IsaacLab `asset.data.joint_acc` is `(current - previous) / dt`.
- `Go2-CTSMoe-v0` now points to `go2_rsl_rl_cts_cfg:Go2CtsRunnerCfg` and uses only the `policy` and `critic` groups required by base CTS.

Next recommended migration block:

- Base CTS, domain randomization, delayed actions, and exporter integration are complete.
- The next block is MoE: first migrate and test the standalone `StudentMoEEncoder`, then integrate `ActorCriticMoENGCTS` and the load-balance loss.

## Domain Randomization Contract

From `GO2Cfg.domain_rand`:

- friction randomization enabled, range `[0.0, 2.0]`
- base mass randomization enabled, added mass `[-1.0, 1.0]`
- link mass randomization enabled, multiplier `[0.9, 1.1]`
- base COM randomization enabled, added range `[-0.03, 0.03]`
- restitution randomization enabled, range `[0.0, 0.5]`
- PD gain randomization enabled:
  - stiffness multiplier `[0.9, 1.1]`
  - damping multiplier `[0.9, 1.1]`
- motor zero offset randomization enabled, range `[-0.035, 0.035]`
- motor strength randomization enabled, range `[0.8, 1.2]`
- random pushes enabled:
  - interval `4` seconds
  - max xy push velocity `0.4`
  - max angular push velocity `0.6`
- action delay randomization enabled:
  - uses last action for a random 0 to 4 decimation steps, i.e. 0 to 20 ms at 50 Hz policy dt

## PPO And CTS Training Config

Plain PPO:

- `GO2CfgPPO.runner.experiment_name = "go2_ppo"`
- `max_iterations = 150000`
- `save_interval = 500`
- actor hidden dims `[512, 256, 128]`
- critic hidden dims `[512, 256, 128]`
- activation `elu`
- PPO: learning rate `1e-3`, gamma `0.99`, lambda `0.95`, entropy coef `0.01`, desired KL `0.01`, epochs `5`, mini-batches `4`, steps per env `24`

CTS:

- `GO2CfgCTS.runner.experiment_name = "go2_cts"`
- `max_iterations = 150000`
- `save_interval = 500`
- `num_steps_per_env = 24`
- `history_length = 5`
- latent dim `32`
- teacher env ratio `0.75`

## Base CTS Migration Complete

Base CTS is migrated end to end and should remain the stable baseline while MoE is added:

- `ActorCriticCTS`: Teacher Encoder uses the 263-dim critic observation; Student Encoder uses the 5-frame policy history; latent dimension is 32.
- `RolloutStorageCTS`: stores history with each transition and preserves Teacher/Student indices and mini-batch ratios.
- `CTS`: PPO optimizer updates Teacher Encoder, actor, critic, and action-noise parameter; a separate optimizer trains Student Encoder with latent imitation.
- `OnPolicyRunnerCTS`: constructs interleaved Teacher/Student env IDs, maintains history in act/step/reset order, runs learning/logging, and saves both optimizers.
- `Go2CtsRunnerCfg`, task registration, `train.py`, and `play.py` select the CTS runner.
- Active dimensions are policy `45`, critic `263`, history `5 x 45 = 225`, latent `32`, actor input `77`, and actions `12`.
- The root environment cfg now includes `commands: CommandsCfg`, so `base_velocity` is registered before observation terms are prepared.

CTS verification:

- The pure PyTorch CTS suite passes 10 tests covering the policy, storage, algorithm, fake-VecEnv runner, and exporter.
- The runner test executes a complete `act -> process_env_step -> compute_returns -> update` and proves both optimizer parameter groups change.
- Save/load tests compare the model and both optimizer state dictionaries exactly.
- Real smoke training completed with 8 envs and 24 steps per env (192 transitions) with finite value, surrogate, entropy, and latent losses.
- `model_0.pt` contains `model_state_dict`, `optimizer_state_dict`, `student_optimizer_state_dict`, `iter`, and `infos`.
- Real play ran 250 steps and produced a video while resetting Student inference history on dones.

CTS-specific export is also complete:

- JIT owns stateful history and exposes `reset()`.
- ONNX is stateless and uses `obs, history -> actions, next_history`.
- The default IsaacLab exporter must not be used for CTS because it exports only the 77-dim actor subnetwork and omits history and Student Encoder.
- Exporter tests compare multiple steps and reset behavior with `ActorCriticCTS.act_inference`; ONNX is executed with `onnx.reference.ReferenceEvaluator` because `onnxruntime` is not installed.

CTS resume follow-up is now resolved:

- `Go2CtsRunnerCfg.load_optimizer = True`, so `train.py` explicitly restores both the PPO optimizer and Student optimizer during resumed training. `OnPolicyRunnerCTS.load()` still defaults to `False`, which keeps `play.py` from loading optimizer state unnecessarily.
- `OnPolicyRunnerCTS.learn()` now stores `current_learning_iteration = it + 1`. After completing iteration 0, a checkpoint stores the next iteration as 1, so resume does not repeat iteration 0.
- The runner dummy test verifies `learn(1) -> iter 1 -> save/load -> learn(1) -> iter 2`.

## MoE Migration Status

The no-goal Student MoE stack is now implemented through task configuration and runner construction while preserving base CTS as a separate working path.

Implemented modules:

- `StudentMoEEncoder` uses a shared no-goal Expert backbone, per-Expert grouped `Conv1d` output heads, a full-history softmax Gate, weighted latent fusion, and L2Norm/SimNorm.
- `ActorCriticMoENGCTS` inherits the validated `ActorCriticCTS` Teacher, actor, critic, action distribution, normalization, and history/reset behavior. It replaces the plain Student Encoder with `student_moe_encoder` and returns both Student latent and Gate weights.
- `MoENGCTS` inherits `CTS`. The PPO optimizer remains responsible for Teacher Encoder, actor, critic, and action noise; the Student optimizer contains the complete Student MoE module.
- The Student loss is `latent_loss + load_balance_coef * load_balance_loss`. Load balance compares the Student mini-batch mean Gate usage against uniform Expert usage.
- `CTS` now exposes `_get_student_encoder_module()` and `_compute_student_loss()` hooks so MoE support does not duplicate the full PPO update.

Active MoE dimensions and mask:

- policy observation `45`, critic observation `263`, history length `5`, latent `32`, actions `12`
- Gate input is the complete `5 x 45 = 225` history
- Expert input removes command indices `6:9` from every frame, giving `5 x 42 = 210`
- the Student has `8` Experts, each with hidden dimension `256`

MoE configuration and routing are connected:

- `RslRlMoENGCTSActorCriticCfg` adds `obs_no_goal_mask`, `student_expert_num`, and `student_expert_hidden_dim`.
- `RslRlMoENGCTSAlgorithmCfg` adds `load_balance_coef`.
- `Go2CtsRunnerCfg` remains the base CTS configuration, while `Go2MoENGCtsRunnerCfg` selects `ActorCriticMoENGCTS + MoENGCTS` and experiment `go2_moe_no_goal_cts`.
- `OnPolicyRunnerCTS._construct_algorithm()` explicitly supports only the valid pairs `ActorCriticCTS + CTS` and `ActorCriticMoENGCTS + MoENGCTS`.
- task `Go2-CTS-v0` selects base CTS; task `Go2-CTSMoe-v0` selects the MoE configuration. Both reuse `CTSMoeRoughEnvCfg`.

MoE verification:

- `StudentMoEEncoder` tests cover input/output dimensions, Gate sums, latent L2 norm, Gate/Expert gradients, and grouped-head channel isolation.
- `ActorCriticMoENGCTS` tests cover framewise no-goal masking, Teacher/Student interfaces and gradient isolation, inference history/reset, and invalid mask length.
- `MoENGCTS` tests execute `act -> process_env_step -> compute_returns -> update`, verify optimizer parameter isolation, and prove PPO, Gate, and Expert parameters all change.
- The current pure PyTorch MoE suite passes `8` tests without Isaac Sim.

Remaining MoE work:

1. Add a fake-VecEnv runner test proving `OnPolicyRunnerCTS` constructs `ActorCriticMoENGCTS + MoENGCTS`, completes `learn(1)`, and saves/loads both optimizer states.
2. Remove the redundant pre-hook `get_student_latent()` call in the CTS Student update if it is still present; `_compute_student_loss()` already performs that forward pass.
3. Resolve the duplicate inactive `agents/go2_rsl_rl_ctsmoe_cfg.py`; task registration currently uses `go2_rsl_rl_cts_cfg.py:Go2MoENGCtsRunnerCfg`.
4. Run an 8-env, 1-iteration Isaac Sim MoE training smoke test and inspect finite `value`, `surrogate`, `entropy`, `latent`, and `load_balance` metrics.
5. Run a 250-step MoE play test.
6. Extend the CTS JIT/ONNX exporter for `student_moe_encoder`, no-goal masking, and Gate output parity. The current CTS exporter still assumes `policy.student_encoder`.

Go2 no-goal mask for the 45-dim policy observation:

- keep indices `0:6` (`base_ang_vel`, `projected_gravity`)
- remove indices `6:9` (`velocity_commands`) from Expert inputs
- keep indices `9:45` (`joint_pos`, `joint_vel`, `actions`)
- Gating still receives the complete 5 x 45 history; Experts receive 5 x 42 history.

## IsaacLab Migration Plan

Recommended staged migration:

1. Create or update `go2_rl_gym_velocity_cfg.py` with the external Go2 actor observation contract: 45 dims, no base linear velocity, no height scan.
2. Add critic/privileged observation group matching the 263-dim external layout as closely as IsaacLab allows: base lin vel, actor obs terms, foot contact forces, torques, accelerations, 187 heights.
3. Match robot init, PD gains, action scale, decimation, episode length, and normalization/noise.
4. Use the existing 187-point IsaacLab ray caster grid.
5. Start with approximate IsaacLab terrain for smoke testing, then migrate terrain proportions and difficulty ranges closer to external `terrain.py`.
6. Port command sampling/curriculum carefully. The external dynamic command sampler is custom and may require extending `mdp/commands.py`.
7. Port terrain curriculum using accumulated xy command logic, not just the default IsaacLab terrain-level update.
8. Port rewards after observation and commands are stable, especially `correct_base_height`, dynamic sigma tracking rewards, `feet_regulation`, and `hip_to_default`.
9. Add or update gym registration in `velocity/__init__.py`.
10. Base CTS is complete. Continue with the MoE sequence documented above.

## Restore Procedure

At the start of any resumed session:

1. Read this `MEMORY.md`.
2. Run `git status --short`.
3. Inspect current files before editing:
   - `source/.../velocity/go2_teacher_student_cfg.py`
   - any `source/.../velocity/go2_rl_gym_velocity_cfg.py`
   - `source/.../velocity/mdp/commands.py`
   - `source/.../velocity/mdp/rewards.py`
   - `source/.../velocity/mdp/curriculums.py`
   - `source/.../velocity/__init__.py`
4. If a value conflicts between this memory and live code, trust live code only after confirming whether the difference was intentional.
