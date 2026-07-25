# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extensions for the learning algorithms."""

from .rnd import RandomNetworkDistillation, resolve_rnd_config
from .symmetry import Symmetry, resolve_symmetry_config
from .amp import AMPDiscriminator, resolve_amp_config
from .actor_critic import ActorCritic
from .actor_critic_cnn import ActorCriticCNN
from .actor_critic_recurrent import ActorCriticRecurrent
from .student_teacher_recurrent import StudentTeacherRecurrent
from .student_teacher import StudentTeacher
from .depth_history_encoder import DepthHistoryEncoder
from .actor_critic_cts import ActorCriticCTS
from .actor_critic_moe_ng_cts import (
    ActorCriticMoENGCTS,
    StudentMoEEncoder,
)
from .actor_critic_dwaq import ActorCriticDWAQ

__all__ = [
    "ActorCritic",
    "ActorCriticCNN",
    "ActorCriticRecurrent",
    "RandomNetworkDistillation",
    "Symmetry",
    "resolve_rnd_config",
    "resolve_symmetry_config",
    "AMPDiscriminator",
    "resolve_amp_config",
    "StudentTeacherRecurrent",
    "DepthHistoryEncoder",
    "ActorCriticCTS",
    "StudentMoEEncoder",
    "ActorCriticMoENGCTS",
    "ActorCriticDWAQ",
]
