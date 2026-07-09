# Isaac Lab Robot Learning Workspace

Blank RL Lab is an Isaac Lab extension workspace for developing, training, and
validating robot learning tasks outside of the core Isaac Lab repository. It is
organized as a project-level lab: task definitions, custom managers, RSL-RL
configuration, motion datasets, utility scripts, and sim-to-sim tools live
together while remaining isolated from the upstream Isaac Lab installation.

The repository is intended for iterative locomotion research and deployment
preparation. It supports manager-based Isaac Lab environments, reinforcement
learning and imitation-learning experiments, policy export, runtime inspection,
and MuJoCo-based replay checks.

## Key Features

- `Isaac Lab extension` Develop local tasks and environment components without
  modifying the Isaac Lab source tree.
- `Manager-based workflows` Compose scene, command, action, observation, event,
  reward, curriculum, and termination logic through Isaac Lab configuration
  classes.
- `RSL-RL integration` Train, play, resume, and export policies with project
  scripts and local runner configuration.
- `Motion-prior tooling` Load reference motion data, initialize robot states
  from motion clips, and train policies with AMP-style discriminator inputs.
- `Teacher-student experiments` Keep privileged training observations separate
  from deployable policy observations.
- `Sim-to-sim validation` Replay exported policies in MuJoCo with explicit
  observation, joint-order, action-scale, and actuator mappings.

## Repository Layout

```text
.
|-- source/blank_rl_lab/                 # Installable Isaac Lab extension
|   |-- blank_rl_lab/
|   |   |-- assets/                      # Robot assets and local asset configs
|   |   |-- datasets/                    # Motion clips and retargeting tools
|   |   |-- envs/                        # Custom manager-based env classes
|   |   |-- managers/                    # Motion and animation managers
|   |   |-- rsl_rl/                      # Project RSL-RL config extensions
|   |   `-- tasks/                       # Registered Isaac Lab tasks
|   |-- config/extension.toml            # Omniverse extension metadata
|   `-- pyproject.toml                   # Python package metadata
|-- scripts/
|   |-- rsl_rl/                          # Train/play/export entry points
|   |-- sim2mujoco/                      # MuJoCo policy replay scaffold
|   |-- list_envs.py                     # Task discovery helper
|   |-- random_agent.py                  # Environment sanity check
|   `-- zero_agent.py                    # Zero-action sanity check
|-- logs/                                # Training logs and exported policies
|-- outputs/                             # Hydra/Isaac Lab runtime outputs
`-- pyproject.toml                       # Workspace tooling configuration
```

## Installation

Install Isaac Lab first by following the official installation guide. A conda or
uv-based Isaac Lab environment is recommended because it makes the project
scripts easier to run from the terminal.

Install this extension in editable mode with the same Python interpreter used by
Isaac Lab:

```bash
# Use /path/to/isaaclab.sh -p instead of python if Isaac Lab is not installed
# directly in the active Python environment.
python -m pip install -e source/blank_rl_lab
```

After installation, verify that the extension is discoverable:

```bash
python scripts/list_envs.py
```

## Common Workflows

### Check an Environment

Use the zero-action and random-action agents before long training runs. These
commands verify that task registration, reset logic, observations, actions, and
simulation stepping are healthy. The examples below use the teacher-student
velocity task.

```bash
python scripts/zero_agent.py --task Go2-ts-velocity-v0
python scripts/random_agent.py --task Go2-ts-velocity-v0
```

### Train a Policy

Training entry points are kept under `scripts/rsl_rl`. Select a registered task
from `scripts/list_envs.py` and pass it to the training script.

```bash
python scripts/rsl_rl/train.py --task Go2-ts-velocity-v0
```

When Isaac Lab is managed by `isaaclab.sh`, run the same script through Isaac
Lab's launcher:

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py --task Go2-ts-velocity-v0
```

### Play or Export a Policy

Use the play script to evaluate a checkpoint and export deployable policy
artifacts when supported by the runner.

```bash
python scripts/rsl_rl/play.py \
    --task Go2-ts-velocity-v0 \
    --checkpoint logs/rsl_rl/go2_demo/<RUN>/model_<ITERATION>.pt
```

Exported artifacts are usually written under the checkpoint run directory, for
example:

```text
logs/rsl_rl/go2_demo/<RUN>/exported/
```

### Inspect Runtime Contracts

When moving a trained policy into another simulator or runtime, inspect the
Isaac Lab runtime contract instead of guessing it from asset files. The important
items are joint order, body names, default joint positions, observation terms,
action scale, action offset, control timestep, and policy decimation.

```bash
./isaaclab.sh -p scripts/sim2mujoco/inspect_isaaclab_task.py \
    --task Go2-ts-velocity-v0 \
    --output /tmp/go2_ts_runtime.yaml \
    --headless
```

### Replay in MuJoCo

The `scripts/sim2mujoco` folder contains an isolated replay scaffold for
sim-to-sim checks. A MuJoCo replay config should explicitly map:

- Isaac Lab policy joint names to MuJoCo joint names.
- Policy action order to MuJoCo actuator names.
- Observation terms to MuJoCo state fields.
- Isaac Lab action scale and default joint offsets to MuJoCo control targets.
- Policy frequency and simulator timestep to MuJoCo stepping.

Start with zero policy or small velocity commands before testing a trained
checkpoint:

```bash
python scripts/sim2mujoco/run.py \
    --config scripts/sim2mujoco/config/go2_ts_velocity.yaml \
    --zero-policy
```

Then run an exported policy:

```bash
python scripts/sim2mujoco/run.py \
    --config scripts/sim2mujoco/config/go2_ts_velocity.yaml \
    --policy logs/rsl_rl/go2_demo/<RUN>/exported/policy.pt
```

## Development Notes

- Keep new tasks under `source/blank_rl_lab/blank_rl_lab/tasks` and register
  them through the task package `__init__.py`.
- Keep environment-level extensions under `envs`, reusable runtime components
  under `managers`, and algorithm configuration under `rsl_rl`.
- Prefer runtime inspection outputs when validating policy transfer. Asset files
  are useful references, but the exported environment contract is the source of
  truth for policy input and action ordering.
- Treat sim-to-sim configs as deployment manifests: every joint name, actuator
  name, observation scale, default offset, and control frequency should be
  written down explicitly.

## Code Formatting

The workspace includes Ruff and Pyright settings in `pyproject.toml`. If
pre-commit is installed, run:

```bash
pre-commit run --all-files
```

You can also run formatter or lint tooling directly from your active Isaac Lab
Python environment.

## IDE Setup

For VS Code, run the Isaac Lab workspace setup task or add the project extension
and Isaac Lab source folders to `python.analysis.extraPaths`. At minimum, the
editor should be able to resolve:

```json
{
    "python.analysis.extraPaths": [
        "<path-to-this-repo>/source/blank_rl_lab",
        "<path-to-isaaclab>/source/isaaclab",
        "<path-to-isaaclab>/source/isaaclab_assets",
        "<path-to-isaaclab>/source/isaaclab_tasks",
        "<path-to-isaaclab>/source/isaaclab_rl"
    ]
}
```

## Acknowledgments

Thanks to [legged_lab](https://github.com/zitongbai/legged_lab) and its author for the valuable reference and inspiration.
