from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

def _env_ids_tensor(env_ids: Sequence[int] | slice | torch.Tensor | None, num_envs: int, device: str) -> torch.Tensor:
    all_env_ids = torch.arange(num_envs, device=device)

    if env_ids is None:
        return all_env_ids

    if isinstance(env_ids, slice):
       return all_env_ids[env_ids]

    return torch.as_tensor(env_ids, dtype=torch.long, device=device)


class Go2DelayedJointPositionAction(JointPositionAction):
    cfg: Go2DelayedJointPositionActionCfg

    def __init__(self, cfg: Go2DelayedJointPositionActionCfg, env):
        super().__init__(cfg, env)

        self._previous_raw_actions = torch.zeros_like(self._raw_actions)
        self._applied_raw_actions = torch.zeros_like(self._raw_actions)
        self._motor_zero_offsets = torch.zeros_like(self._raw_actions)
        self._delay_steps = torch.zeros(self.num_envs, 1, dtype=torch.long, device=self.device)
        self._apply_step = 0

    def process_actions(self, actions: torch.Tensor):
        self._previous_raw_actions[:] = self._raw_actions
        self._raw_actions[:] = actions
        self._apply_step = 0

        if self.cfg.randomize_action_delay:
            self._delay_steps[:] = torch.randint(
                low=0,
                high=self.cfg.max_delay_steps + 1,
                size=(self.num_envs, 1),
                device=self.device,
            )
        else:
            self._delay_steps.zero_()

        self._processed_actions[:] = self._process_raw_actions(self._raw_actions)

    def apply_actions(self):
        if self.cfg.randomize_action_delay:
            use_current = self._apply_step >= self._delay_steps
            raw_actions = torch.where(use_current, self._raw_actions, self._previous_raw_actions)
        else:
            raw_actions = self._raw_actions

        self._applied_raw_actions[:] = raw_actions
        self._processed_actions[:] = self._process_raw_actions(raw_actions)
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

        self._apply_step += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids_tensor = _env_ids_tensor(env_ids, self.num_envs, self.device)

        self._raw_actions[env_ids_tensor] = 0.0
        self._previous_raw_actions[env_ids_tensor] = 0.0
        self._applied_raw_actions[env_ids_tensor] = 0.0
        self._delay_steps[env_ids_tensor] = 0

        self._sample_motor_zero_offsets(env_ids_tensor)
        self._processed_actions[env_ids_tensor] = self._process_raw_actions(self._raw_actions)[env_ids_tensor]

    def _sample_motor_zero_offsets(self, env_ids: Sequence[int] | slice | torch.Tensor):
        env_ids_tensor = _env_ids_tensor(env_ids, self.num_envs, self.device) # type:ignore

        if not self.cfg.randomize_motor_zero_offset:
           self._motor_zero_offsets[env_ids_tensor] = 0.0
           return

        low, high = self.cfg.motor_zero_offset_range
        self._motor_zero_offsets[env_ids_tensor] = torch.empty(
            len(env_ids_tensor),
            self.action_dim,
            device=self.device,
        ).uniform_(low, high)

    def _process_raw_actions(self, raw_actions: torch.Tensor) -> torch.Tensor:
        processed_actions = raw_actions * self._scale + self._offset
        processed_actions = processed_actions + self._motor_zero_offsets

        if self.cfg.clip is not None:
            processed_actions = torch.clamp(
                processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

        return processed_actions



@configclass
class Go2DelayedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = Go2DelayedJointPositionAction

    randomize_motor_zero_offset: bool = True
    motor_zero_offset_range: tuple[float, float] = (-0.035, 0.035)

    randomize_action_delay: bool = True
    max_delay_steps: int = 4

class Go2MotorStrengthIdealPDActuator(IdealPDActuator):
    cfg: Go2MotorStrengthIdealPDActuatorCfg

    def __init__(self, cfg: Go2MotorStrengthIdealPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.motor_strengths = torch.ones_like(self.applied_effort)

    def reset(self, env_ids: Sequence[int] | None):
        super().reset(env_ids) # type:ignore

        env_ids_tensor = _env_ids_tensor(env_ids, self._num_envs, self._device)

        if not self.cfg.randomize_motor_strength:
            self.motor_strengths[env_ids_tensor] = 1.0
            return

        low, high = self.cfg.motor_strength_range
        self.motor_strengths[env_ids_tensor] = torch.empty(
            len(env_ids_tensor),
            self.num_joints,
            device=self._device,
        ).uniform_(low, high)


    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
       control_action = super().compute(control_action, joint_pos, joint_vel)

       self.applied_effort *= self.motor_strengths
       control_action.joint_efforts = self.applied_effort

       return control_action


@configclass
class Go2MotorStrengthIdealPDActuatorCfg(IdealPDActuatorCfg):
    class_type: type = Go2MotorStrengthIdealPDActuator

    randomize_motor_strength: bool = True
    motor_strength_range: tuple[float, float] = (0.8, 1.2)