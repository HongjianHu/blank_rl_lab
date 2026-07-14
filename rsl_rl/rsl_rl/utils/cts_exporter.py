from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from rsl_rl.modules import ActorCriticCTS

class _CtsPolicyCore(nn.Module):
    """Stateless CTS inference: (obs, history) -> (actions, next_history)."""

    def __init__(self, policy: ActorCriticCTS) -> None:
        super().__init__()
        self.normalizer = copy.deepcopy(policy.actor_obs_normalizer).cpu()
        self.student_encoder = copy.deepcopy(policy.student_encoder).cpu()
        self.actor = copy.deepcopy(policy.actor).cpu()

        self.history_length = policy.history_length
        self.num_actor_obs = policy.num_actor_obs

    def forward(self, obs: torch.Tensor, history: torch.Tensor,) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_obs = self.normalizer(obs)

        next_history = torch.cat((history[:,1:], normalized_obs.unsqueeze(1)), dim=1)

        latent = self.student_encoder(next_history.flatten(start_dim=1))
        actor_input = torch.cat((normalized_obs, latent), dim=-1)
        actions = self.actor(actor_input)

        return actions, next_history

class _CtsJitPolicy(nn.Module):
    """Stateful JIT policy that owns its observation history."""

    def __init__(self, policy:ActorCriticCTS, num_envs:int = 1) -> None:
        super().__init__()
        self.core = _CtsPolicyCore(policy)
        self.register_buffer(
            "history",
            torch.zeros(
                num_envs,
                policy.history_length,
                policy.num_actor_obs,
            )
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        actions, next_history = self.core(obs, self.history)
        self.history.copy_(next_history)
        return actions
    # forward() 会自动成为 JIT 入口，但其他方法默认不一定被导出，因此用：@torch.jit.export 显式保留 reset()。
    @torch.jit.export
    def reset(self) -> None:
        self.history.zero_()

def export_cts_policy_as_jit(
    policy: ActorCriticCTS,
    path: str,
    filename: str = "policy.pt",
    num_envs: int = 1,
) -> None:
    os.makedirs(path, exist_ok=True)
    # 网络切换到推理模式，避免 Dropout、BatchNorm 等模块保持训练行为
    exporter = _CtsJitPolicy(policy, num_envs=num_envs).cpu().eval()
    # 这里使用 script，不是 trace, 原因是模型包含：内部history修改,reset方法,可变状态
    # torch.jit.script() 会分析完整代码逻辑，而 trace() 主要记录一次样例运行经过的算子
    scripted = torch.jit.script(exporter)
    # 最终文件包含:Normalizer,Student Encoder,Actor,内部history,reset方法
    scripted.save(os.path.join(path, filename))

def export_cts_policy_as_onnx(
    policy: ActorCriticCTS,
    path: str,
    filename: str = "policy.onnx",
) -> None:
    os.makedirs(path, exist_ok=True)

    exporter = _CtsPolicyCore(policy).cpu().eval() # 它没有内部 history，因此输入必须包含：obs, history
    obs = torch.zeros(1, policy.num_actor_obs)
    history = torch.zeros(
        1,
        policy.history_length,
        policy.num_actor_obs,
    )

    # obs = torch.zeros(1, 45), history = torch.zeros(1, 5, 45)
    # 这些样例主要用于告诉导出器输入数量、形状和数据类型。
    # input_names=["obs", "history"], output_names=["actions", "next_history"]
    # 让 ONNX 接口具有可读名称，而不是自动生成的名称。
    torch.onnx.export(
        exporter,
        (obs, history),
        os.path.join(path, filename),
        export_params=True,
        opset_version=18,
        input_names=["obs", "history"],
        output_names=["actions", "next_history"],
        dynamic_axes={ # 只允许第0维 batch 动态变化：
            "obs": {0: "batch"},
            "history": {0: "batch"},
            "actions": {0: "batch"},
            "next_history": {0: "batch"},
        },
    )
