# IsaacLab Policy to MuJoCo Runtime

This folder is a teaching-oriented sim-to-MuJoCo scaffold. It is intentionally
split into small files so each step in the policy replay pipeline can be checked
and understood independently.

## What This Runner Does

The main loop in `run.py` follows this order:

1. Read MuJoCo `qpos/qvel` through `MujocoRobot`.
2. Convert state into policy-ordered joint arrays.
3. Build the IsaacLab policy observation in `observation.py`.
4. Run the exported policy from `policy.py`.
5. Convert action into joint position targets in `control.py`.
6. Apply PD torque or position actuator commands.
7. Step MuJoCo and update the viewer.

The example runtime config is `go2_ts_velocity`, matching the deployable policy
group in `Go2-ts-velocity-v0`:

```text
base_ang_vel * 0.2
projected_gravity
velocity_commands
joint_pos_rel
joint_vel_rel * 0.05
joint_effort * 0.01
last_action
```

For a 12-DoF Go2 this is 57 dimensions.

## Files

- `config/go2_ts_velocity.yaml`: editable Go2 teacher-student replay config.
- `run.py`: the MuJoCo rollout loop.
- `inspect_isaaclab_task.py`: helper for dumping IsaacLab runtime joint order.
- `sim2mujoco/config.py`: YAML parsing and validation.
- `sim2mujoco/mujoco_adapter.py`: MuJoCo model/data boundary and joint mapping.
- `sim2mujoco/observation.py`: IsaacLab observation reconstruction.
- `sim2mujoco/control.py`: action scale, default offset, and PD control.
- `sim2mujoco/policy.py`: TorchScript/ONNX/zero-policy inference wrappers.
- `sim2mujoco/viewer.py`: passive viewer setup and policy command keys.

## Step 1: Export the IsaacLab Runtime Order

Use this before writing a final mapping:

```bash
./isaaclab.sh -p scripts/sim2mujoco/inspect_isaaclab_task.py \
    --task Go2-ts-velocity-v0 \
    --output /tmp/go2_ts_runtime.yaml \
    --headless
```

Use the dumped action term `joint_names` as the policy action order. Use
`robot.default_joint_pos` as the default offset.

## Step 2: Prepare a MuJoCo XML

This repository currently contains Go2 USD assets, not a MuJoCo XML. Point the
config at a valid MuJoCo model:

```yaml
robot:
  model_path: "/absolute/path/to/go2/scene.xml"
```

Then print MuJoCo names:

```bash
python3 scripts/sim2mujoco/run.py \
    --config scripts/sim2mujoco/config/go2_ts_velocity.yaml \
    --model /absolute/path/to/go2/scene.xml \
    --print-model-names
```

Use that output to fill `mujoco_joint_names` and, if needed,
`mujoco_actuator_names`.

## Step 3: Debug Without the Policy

Zero action should hold the default pose:

```bash
python3 scripts/sim2mujoco/run.py \
    --config scripts/sim2mujoco/config/go2_ts_velocity.yaml \
    --model /absolute/path/to/go2/scene.xml \
    --zero-policy
```

If the robot falls immediately here, debug the MuJoCo model, default pose,
actuator mode, PD gains, or joint mapping before loading the neural policy.

## Step 4: Run an Exported Policy

```bash
python3 scripts/sim2mujoco/run.py \
    --config scripts/sim2mujoco/config/go2_ts_velocity.yaml \
    --model /absolute/path/to/go2/scene.xml \
    --policy logs/rsl_rl/go2_demo/<run>/exported/policy.pt \
    --command 0.2 0.0 0.0
```

## Viewer Control

Use MuJoCo's native viewer UI, mouse controls, and built-in visualization
shortcuts for camera movement, pause/reset controls, and debug overlays.

The sim2mujoco callback only handles policy velocity commands:

```text
up/down: increase/decrease vx
left/right: increase/decrease yaw rate wz
page up/page down: increase/decrease lateral velocity vy
backspace: zero velocity command
```

## Implementation Order For Learning

1. Verify IsaacLab joint/action order with `inspect_isaaclab_task.py`.
2. Verify MuJoCo joint/actuator names with `--print-model-names`.
3. Run `--zero-policy` and make the robot stand.
4. Print one MuJoCo observation and compare with IsaacLab for the same pose.
5. Load `policy.pt` and start with small commands like `--command 0.1 0 0`.
6. Add diagnostics: base height, roll/pitch, action norm, torque norm, contacts.
