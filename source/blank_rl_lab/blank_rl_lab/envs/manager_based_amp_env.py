from __future__ import annotations

import torch
from typing import Any
from collections.abc import Sequence
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv, VecEnvStepReturn, VecEnvObs
from isaaclab.managers import ActionManager, ObservationManager, RecorderManager, CommandManager, CurriculumManager, RewardManager, TerminationManager

from blank_rl_lab.managers import MotionDataManager, AnimationManager
from .manager_based_animation_env import ManagerBasedAnimationEnv
from .manager_based_amp_env_cfg import ManagerBasedAmpEnvCfg

class ManagerBasedAmpEnv(ManagerBasedAnimationEnv):

    """AMP Environment for locomotion tasks.

    This class inherits from the `ManagerBasedRLEnv` class and is used to create an environment for
    training and testing reinforcement learning agents on locomotion tasks using the AMP.

    In the original ManagerBasedRLEnv's `step` method, observations are lost if the environments are
    reset. But in AMP we should record the observations before resetting the environments.
    This class overrides the `step` method to ensure that observations are retained even when
    environments are reset.
    """

    cfg: ManagerBasedAmpEnvCfg

    def __init__(self, cfg: ManagerBasedAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    # def _get_amp_observations(self) -> torch.Tensor:
    #     """Get the AMP observations.

    #     This function retrieves the AMP observations from the observation manager and returns them.

    #     Returns:
    #         The AMP observations as a tensor.
    #     """

    #     amp_obs = self.observation_manager.compute_group("amp", update_history=False)
    #     return amp_obs # (num_envs, history_length, obs_dim)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        This function is almost identical to the parent class, except that:
            In the parent class's method, the observations are computed after the reset, which leads to
            the loss of observations for the reset environments. In this class, we compute the AMP observations
            before the reset and update the observations for the reset environments.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
            The AMP observations are included in the observations dictionary under the key "amp".
        """
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update animation manager
        self.animation_manager.update(dt=self.step_dt)
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        # # -- update AMP observations
        # amp_obs = self._get_amp_observations()

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.extras["terminal_obs"] = self._compute_terminal_disc_obs()
        else:
            self.extras.pop("terminal_obs", None)

        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids) # type: ignore

            self._reset_idx(reset_env_ids) # type: ignore
            # update articulation kinematics
            self.scene.write_data_to_sim() # 将重置后的最新位置、姿态显式写入仿真器
            self.sim.forward()             # 强制向前推算运动学（更新关节位置、速度等 Buffer）

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids) # type: ignore

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)
        # if len(reset_env_ids) > 0:
        #     self.obs_buf["amp"][reset_env_ids] = amp_obs[reset_env_ids]

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _compute_terminal_disc_obs(self) -> dict[str, torch.Tensor]:
        """Build terminal discriminator observations without advancing history.

        The AMP discriminator consumes only the ``disc`` group from terminal_obs.
        We manually append the current post-physics AMP state to the existing
        history so reset environments use ``(s_t, s_terminal)`` instead of the
        post-reset observation.
        """
        group_name = "disc"
        if group_name not in self.observation_manager._group_obs_term_names:
            return {}

        current_disc_obs = self._compute_current_disc_terms(group_name)
        disc_obs = self.observation_manager.compute_group(group_name, update_history=False)
        if not isinstance(disc_obs, torch.Tensor) or disc_obs.ndim != 3:
            return {group_name: current_disc_obs.unsqueeze(1)}

        terminal_disc_obs = torch.roll(disc_obs.clone(), shifts=-1, dims=1)
        terminal_disc_obs[:, -1, :] = current_disc_obs
        return {group_name: terminal_disc_obs}

    def _compute_current_disc_terms(self, group_name: str) -> torch.Tensor:
        term_names = self.observation_manager._group_obs_term_names[group_name]
        term_cfgs = self.observation_manager._group_obs_term_cfgs[group_name]
        obs_terms = []
        for term_cfg in term_cfgs:
            obs = term_cfg.func(self, **term_cfg.params).clone()
            if term_cfg.modifiers is not None:
                for modifier in term_cfg.modifiers:
                    obs = modifier.func(obs, **modifier.params)
            if term_cfg.clip:
                obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
            if term_cfg.scale is not None:
                obs = obs.mul_(term_cfg.scale)
            obs_terms.append(obs)

        if self.observation_manager._group_obs_concatenate[group_name]:
            concat_dim = self.observation_manager._group_obs_concatenate_dim[group_name]
            return torch.cat(obs_terms, dim=concat_dim)
        return torch.cat([obs_terms[i] for i, _ in enumerate(term_names)], dim=-1)
