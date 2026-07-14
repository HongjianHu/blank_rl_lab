from __future__ import annotations

import torch

from rsl_rl.modules.actor_critic_moe_ng_cts import StudentMoEEncoder


BATCH_SIZE = 6
EXPERT_DIM = 210
GATING_DIM = 225
EXPERT_NUM = 8
EXPERT_HIDDEN_DIM = 16
LATENT_DIM = 32


def make_encoder() -> StudentMoEEncoder:
    return StudentMoEEncoder(
        expert_dim=EXPERT_DIM,
        gating_dim=GATING_DIM,
        hidden_dims=[32, 16],
        expert_num=EXPERT_NUM,
        expert_hidden_dim=EXPERT_HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        norm_type="l2norm",
    )


def has_nonzero_grad(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in module.parameters()
    )


def test_output_shapes_gate_sum_and_l2_norm() -> None:
    torch.manual_seed(0)
    encoder = make_encoder()
    full_history = torch.randn(BATCH_SIZE, GATING_DIM)
    no_goal_history = torch.randn(BATCH_SIZE, EXPERT_DIM)

    latent, weights = encoder(full_history, no_goal_history)

    assert latent.shape == (BATCH_SIZE, LATENT_DIM)
    assert weights.shape == (BATCH_SIZE, EXPERT_NUM)
    torch.testing.assert_close(
        weights.sum(dim=-1),
        torch.ones(BATCH_SIZE),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        latent.norm(p=2, dim=-1),
        torch.ones(BATCH_SIZE),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_gradients_reach_gate_and_expert_paths() -> None:
    torch.manual_seed(1)
    encoder = make_encoder()
    full_history = torch.randn(BATCH_SIZE, GATING_DIM)
    no_goal_history = torch.randn(BATCH_SIZE, EXPERT_DIM)

    latent, weights = encoder(full_history, no_goal_history)
    latent_target = torch.randn_like(latent)
    loss = (latent * latent_target).sum() + weights.square().sum()
    loss.backward()

    assert has_nonzero_grad(encoder.gating_network)
    assert has_nonzero_grad(encoder.experts_backbone)
    assert has_nonzero_grad(encoder.experts_hidden)
    assert has_nonzero_grad(encoder.experts_out)


def test_grouped_expert_heads_do_not_mix_channels() -> None:
    encoder = make_encoder()

    with torch.no_grad():
        encoder.experts_out.weight.fill_(1.0)
        encoder.experts_out.bias.zero_()

    for expert_id in range(EXPERT_NUM):
        grouped_input = torch.zeros(
            1,
            EXPERT_NUM * EXPERT_HIDDEN_DIM,
            1,
        )
        start = expert_id * EXPERT_HIDDEN_DIM
        end = start + EXPERT_HIDDEN_DIM
        grouped_input[:, start:end] = 1.0

        grouped_output = encoder.experts_out(grouped_input)
        grouped_output = grouped_output.reshape(
            1,
            EXPERT_NUM,
            LATENT_DIM,
        )

        expected = torch.zeros_like(grouped_output)
        expected[:, expert_id] = float(EXPERT_HIDDEN_DIM)
        torch.testing.assert_close(grouped_output, expected)
