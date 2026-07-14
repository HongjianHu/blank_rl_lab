from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import ActorCriticMoENGCTS
from rsl_rl.storage import RolloutStorageCTS

from .cts import CTS

class MoENGCTS(CTS):
    """CTS with a no-goal Student Mixture-of-Experts encoder."""

    policy: ActorCriticMoENGCTS

    def __init__(
        self,
        policy: ActorCriticMoENGCTS,
        storage: RolloutStorageCTS,
        load_balance_coef: float = 0.01,
        **kwargs: Any,
    ) -> None:
        if load_balance_coef < 0.0:
            raise ValueError(
                "load_balance_coef must be non-negative."
            )

        self.load_balance_coef = load_balance_coef

        super().__init__(policy=policy, storage=storage, **kwargs)

    def _get_student_encoder_module(self) -> nn.Module:
        """Return the Student module optimized by latent imitation."""
        return self.policy.student_moe_encoder

    def _compute_student_loss(self, student_obs:TensorDict, student_history:torch.Tensor)->tuple[torch.Tensor, dict[str, torch.Tensor]]:
        student_latent, gating_weights = self.policy.get_student_latent_and_weights(student_history)

        with torch.no_grad():
            teacher_latent = self.policy.get_teacher_latent(student_obs)

        latent_loss = torch.nn.functional.mse_loss(
            student_latent,
            teacher_latent,
        )
        # 计算每个Expert在整个mini-batch中的平均使用率
        mean_usage = gating_weights.mean(dim=0)

        target_usage = torch.full_like(mean_usage, 1.0 / gating_weights.shape[-1])

        load_balance_loss = (mean_usage - target_usage).square().mean()

        student_loss = latent_loss + self.load_balance_coef * load_balance_loss

        return student_loss, {"latent": latent_loss, "load_balance": load_balance_loss,}
