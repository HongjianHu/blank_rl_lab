# Blank RL Lab

> **Learning-based locomotion for Unitree Go2 and Go2W**
>
> An Isaac Lab workspace for quadruped and wheel-legged locomotion, covering task design, multi-algorithm training, privileged-information distillation, visual and history-based adaptation, and MuJoCo sim-to-sim validation.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-Extension-76B900?logo=nvidia&logoColor=white)](https://github.com/isaac-sim/IsaacLab)
[![PyTorch](https://img.shields.io/badge/PyTorch-RL-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![RSL-RL](https://img.shields.io/badge/RSL--RL-Customized-8A2BE2)](./rsl_rl)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-Sim--to--Sim-1E88E5)](https://mujoco.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux-FCC624?logo=linux&logoColor=black)](https://www.kernel.org/)

[Tasks and Algorithms](#tasks-and-algorithms) · [Quick Start](#quick-start) · [Sim-to-Sim](#sim-to-sim-validation) · [Repository Structure](#repository-structure)

<p align="center">
  <img src="./docs/media/go2_distillation_demo.gif" width="760" alt="Unitree Go2 teacher-student policy training on mixed rough terrain">
</p>
<p align="center"><sub>Go2 teacher-student policy training and debugging across parallel mixed-terrain environments</sub></p>

## Overview

Blank RL Lab studies robust velocity tracking for the **Unitree Go2 quadruped** and **Go2W wheel-legged robot** on flat ground, procedurally generated rough terrain, and obstacle-rich environments. Instead of stopping at a single PPO baseline, the project builds an extensible workflow from task formulation to policy deployment:

1. Define scenes, commands, observations, actions, rewards, randomization, curricula, and termination conditions with the Isaac Lab Manager-Based API.
2. Implement and integrate locomotion algorithms, network modules, runners, and rollout storage in a locally maintained RSL-RL fork.
3. Use privileged observations, depth images, proprioceptive history, and motion priors to improve adaptation to terrain and dynamics variations.
4. Export policies to TorchScript or ONNX and reconstruct their observation and control interfaces explicitly for MuJoCo sim-to-sim validation.

## Tasks and Algorithms

### Registered Environments

| Robot | Task ID | Terrain | Method / Purpose |
|---|---|---|---|
| Go2 | `Go2-velocity-v0` | Flat | PPO velocity-tracking baseline |
| Go2 | `Go2-rough-velocity-v0` | Rough | PPO with terrain curriculum; used as the rough-terrain teacher |
| Go2 | `Go2-ts-velocity-v0` | Rough | Distillation from a privileged teacher to a recurrent GRU student |
| Go2 | `Go2-AMP-velocity-v0` | Flat | PPO with Adversarial Motion Priors |
| Go2 | `Go2-AMP-rough-velocity-v0` | Rough | AMP extended to rough-terrain locomotion |
| Go2 | `Go2-TSDepth-v0` | Rough + Depth | CNN-GRU depth-history encoding with teacher-student training |
| Go2 | `Go2-CTS-v0` | Rough | Concurrent Teacher-Student latent imitation |
| Go2 | `Go2-CTSMoe-v0` | Rough | CTS with a Mixture-of-Experts encoder and load balancing |
| Go2 | `Go2-DWAQ-v0` | Flat | PPO with a beta-VAE for velocity estimation and context inference |
| Go2W | `Go2W-flat-velocity-v0` | Flat | PPO baseline with a hybrid wheel-legged action space |
| Go2W | `Go2W-rough-velocity-v0` | Rough | Rough-terrain velocity tracking for the wheel-legged platform |

### Method Summary

| Method | Deployable Policy Input | Training Objective |
|---|---|---|
| PPO | Current proprioception and velocity command | Flat- and rough-terrain locomotion baseline |
| Distillation | Deployable observations with GRU history | Imitate a teacher that has access to privileged observations |
| AMP | Proprioception and motion-reference data | Optimize task rewards together with a discriminator-based style reward |
| TSDepth | Proprioception and depth history | Reconstruct the privileged terrain latent from visual temporal features |
| CTS | Proprioceptive history; privileged input for the teacher | Train teacher and student concurrently in the same vectorized rollout |
| CTS-MoE | Proprioceptive history and MoE latent | Learn expert specialization, gating, and balanced expert utilization |
| DWAQ | Proprioceptive observation history | Jointly learn velocity estimation, observation reconstruction, and a context latent with a beta-VAE |

## Quick Start

### 1. Requirements

- Linux
- NVIDIA GPU with a compatible driver
- Python 3.10 or 3.11
- NVIDIA Isaac Sim and Isaac Lab

Install Isaac Lab first by following the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). Use the same Python environment for Isaac Lab and this project.

### 2. Install the Project

```bash
git clone https://github.com/HongjianHu/blank_rl_lab.git
cd blank_rl_lab

# This repository contains a customized RSL-RL implementation.
python -m pip install -e rsl_rl
python -m pip install -e source/blank_rl_lab
```

If Isaac Lab's Python is not directly available in the current shell, replace `python` in the following commands with:

```bash
<path-to-IsaacLab>/isaaclab.sh -p
```

Verify that the extension and tasks are registered:

```bash
python scripts/list_envs.py
```

### 3. Environment Smoke Test

Before starting a long training run, use zero and random actions to verify task registration, resets, observations, actions, and simulation stepping:

```bash
python scripts/zero_agent.py --task Go2-rough-velocity-v0
python scripts/random_agent.py --task Go2-rough-velocity-v0
```

### 4. Train a Policy

Train the Go2 rough-terrain PPO baseline:

```bash
python scripts/rsl_rl/train.py \
    --task Go2-rough-velocity-v0 \
    --headless
```

Train DWAQ:

```bash
python scripts/rsl_rl/train.py \
    --task Go2-DWAQ-v0 \
    --headless
```

Train a Go2W rough-terrain policy:

```bash
python scripts/rsl_rl/train.py \
    --task Go2W-rough-velocity-v0 \
    --headless
```

Teacher-student distillation requires a teacher checkpoint:

```bash
python scripts/rsl_rl/train.py \
    --task Go2-ts-velocity-v0 \
    --load_run <TEACHER_RUN> \
    --checkpoint <TEACHER_CHECKPOINT> \
    --headless
```

Common options:

```text
--num_envs <N>              Number of parallel environments
--max_iterations <N>        Maximum number of training iterations
--seed <N>                  Random seed
--video                     Record training videos
--resume                    Resume training from a checkpoint
--load_run <RUN>            Select an experiment run
--checkpoint <FILE>         Select a checkpoint file
```

Training artifacts are stored under:

```text
logs/rsl_rl/<experiment_name>/<run>/
├── model_<iteration>.pt
├── params/
└── videos/                 # Generated when --video is enabled
```

### 5. Evaluate and Export

```bash
python scripts/rsl_rl/play.py \
    --task Go2-rough-velocity-v0 \
    --load_run <RUN> \
    --checkpoint <CHECKPOINT>
```

When the policy architecture is compatible, `play.py` exports both TorchScript and ONNX models:

```text
logs/rsl_rl/<experiment_name>/<run>/exported/
├── policy.pt
└── policy.onnx
```

## Sim-to-Sim Validation

This project treats sim-to-sim transfer as an explicit runtime-interface problem rather than simply loading the same network in another simulator:

```text
MuJoCo qpos/qvel
  → reorder joint states into policy order
  → reconstruct the Isaac Lab observation
  → run TorchScript / ONNX policy inference
  → apply action scale and default joint offsets
  → compute joint targets
  → apply PD torque or position-actuator commands
  → step MuJoCo
```

### Export the Isaac Lab Runtime Contract

```bash
<path-to-IsaacLab>/isaaclab.sh -p \
    scripts/sim2mujoco/inspect_isaaclab_task.py \
    --task Go2-ts-velocity-v0 \
    --output /tmp/go2_ts_runtime.yaml \
    --headless
```

### Validate a Zero Policy First

```bash
python scripts/sim2mujoco/run.py \
    --config scripts/sim2mujoco/config/go2_ts_velocity.yaml \
    --zero-policy
```

### Replay an Exported Policy

```bash
python scripts/sim2mujoco/run.py \
    --config scripts/sim2mujoco/config/go2_ts_velocity.yaml \
    --policy logs/rsl_rl/go2_demo/<RUN>/exported/policy.pt \
    --command 0.2 0.0 0.0
```

Reference configurations:

- `go2_ts_velocity.yaml`: 57-dimensional deployable teacher-student observation.
- `go2_rough_velocity.yaml`: rough-terrain policy with a height scan.
- `go2_amp_template.yaml`: explicit mapping template for an AMP policy.

See [`scripts/sim2mujoco/README.md`](./scripts/sim2mujoco/README.md) for joint ordering, actuator mapping, control timing, and debugging details.

## Repository Structure

```text
.
├── source/blank_rl_lab/
│   └── blank_rl_lab/
│       ├── assets/                      # Go2, Go2W, and other robot assets
│       ├── envs/                        # Custom AMP and animation environments
│       ├── managers/                    # Motion-data and animation managers
│       ├── rsl_rl/                      # Project-side RSL-RL configuration classes
│       └── tasks/manager_based/
│           └── locomotion/
│               ├── legged/velocity/     # Go2 tasks, MDP terms, and agent configs
│               └── wheeled/velocity/    # Go2W tasks, MDP terms, and agent configs
├── rsl_rl/
│   └── rsl_rl/
│       ├── algorithms/                  # PPO, AMP, CTS, TSDepth, and DWAQ
│       ├── modules/                     # Actor-critic, MoE, VAE, and visual encoders
│       ├── runners/                     # Training loops and checkpoint management
│       └── storage/                     # Rollout and algorithm-specific buffers
├── scripts/
│   ├── rsl_rl/                          # Training and evaluation entry points
│   ├── sim2mujoco/                      # MuJoCo replay for exported policies
│   ├── list_envs.py
│   ├── zero_agent.py
│   └── random_agent.py
└── pyproject.toml                       # Ruff, Pyright, and pytest configuration
```

## Suggested Code Walkthrough

For a quick technical walkthrough, read the project in the following order:

1. [`velocity/__init__.py`](./source/blank_rl_lab/blank_rl_lab/tasks/manager_based/locomotion/legged/velocity/__init__.py): Gym task registration and environment-agent bindings.
2. [`go2_rough_velocity.py`](./source/blank_rl_lab/blank_rl_lab/tasks/manager_based/locomotion/legged/velocity/go2_rough_velocity.py): Go2 rough-terrain environment configuration.
3. [`mdp/`](./source/blank_rl_lab/blank_rl_lab/tasks/manager_based/locomotion/legged/velocity/mdp): commands, observations, actions, rewards, curricula, and events.
4. [`rsl_rl/rsl_rl/algorithms/`](./rsl_rl/rsl_rl/algorithms): loss functions, optimizers, and update procedures for each algorithm.
5. [`scripts/rsl_rl/train.py`](./scripts/rsl_rl/train.py): environment creation, runner selection, checkpoint resume, and the training entry point.
6. [`scripts/sim2mujoco/`](./scripts/sim2mujoco): policy input-output contracts and the cross-simulator control loop.

## Project Status and Roadmap

- [x] Go2 flat- and rough-terrain PPO baselines
- [x] Teacher-student distillation with a recurrent student
- [x] AMP motion-prior training
- [x] TSDepth, CTS, CTS-MoE, and DWAQ integration
- [x] Go2W flat- and rough-terrain tasks
- [x] TorchScript / ONNX export and MuJoCo replay scaffold
- [ ] Add, register, and systematically train Extreme Parkour environments
- [ ] Add a unified evaluation protocol, training curves, and multi-seed comparisons
- [ ] Benchmark velocity tracking, stability, and terrain traversal across algorithms
- [ ] Extend validation toward sim-to-real interfaces and hardware safety constraints

## Acknowledgments

This project builds on and draws inspiration from the following open-source projects:

- [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
- [legged_lab](https://github.com/zitongbai/legged_lab)
- [robot_lab](https://github.com/fan-ziqi/robot_lab)

Thanks to their authors and communities for making these resources available. Third-party assets and derived code in this repository remain subject to their respective original licenses.
