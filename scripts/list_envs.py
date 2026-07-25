# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the available environments in Isaac Lab.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.

All project environments are registered by the `blank_rl_lab.tasks` package.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="List Isaac Lab environments.")
parser.add_argument("--keyword", type=str, default=None, help="Keyword to filter environments.")
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

import blank_rl_lab.tasks  # noqa: F401


def main():
    """Print all environments registered by the ``blank_rl_lab`` project."""
    # print all the available environments
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Isaac Lab"
    # set alignment of table columns
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    # count of environments
    index = 0
    # Acquire all environments whose configuration is provided by this project.
    keyword = args_cli.keyword.lower() if args_cli.keyword is not None else None
    for task_spec in sorted(gym.registry.values(), key=lambda spec: spec.id):
        env_cfg_entry_point = task_spec.kwargs.get("env_cfg_entry_point", "")
        is_project_task = isinstance(env_cfg_entry_point, str) and env_cfg_entry_point.startswith("blank_rl_lab.")
        matches_keyword = keyword is None or keyword in task_spec.id.lower()
        if not (is_project_task and matches_keyword):
            continue

        table.add_row([index + 1, task_spec.id, task_spec.entry_point, env_cfg_entry_point])
        index += 1

    print(table)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # close the app
        simulation_app.close()
