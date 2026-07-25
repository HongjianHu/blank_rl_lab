from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from rsl_rl.networks import MLP

@dataclass
class ContextVAEOutput:

    code: torch.Tensor
    """Full latent code ``[velocity | latent]``  — shape ``(B, code_dim)``."""

    code_vel: torch.Tensor
    """Reparameterised velocity estimate — shape ``(B, velocity_dim)``."""

    code_latent: torch.Tensor
    """Reparameterised latent state — shape ``(B, code_dim - velocity_dim)``."""

    reconstruction: torch.Tensor
    """Decoder reconstruction of the policy observation — shape ``(B, output_dim)``."""

    mean_vel: torch.Tensor
    """Velocity branch posterior mean — shape ``(B, velocity_dim)``."""

    logvar_vel: torch.Tensor
    """Velocity branch posterior log-variance — shape ``(B, velocity_dim)``."""

    mean_latent: torch.Tensor
    """Latent branch posterior mean — shape ``(B, code_dim - velocity_dim)``."""

    logvar_latent: torch.Tensor
    """Latent branch posterior log-variance — shape ``(B, code_dim - velocity_dim)``."""

class ContextVAE(nn.Module):
    """β-VAE context encoder for DreamWAQ.

    Architecture::

        obs_history ──► Encoder MLP ──► h
                                         ├──► mean_vel / logvar_vel   ──► code_vel   ─┐
                                         └──► mean_latent / logvar_latent ──► code_latent ─┤
                                                                                           │
                                                         code = [code_vel | code_latent] ◄─┘
                                                                       │
                                                               Decoder MLP ──► reconstruction

    The encoder maps a flattened observation history to an intermediate hidden
    representation, which is then projected into two independent Gaussian
    branches (velocity and latent). Samples from both branches are concatenated
    to form the full latent *code*, which the decoder maps back to a
    reconstruction of the current policy observation.

    During training the VAE loss consists of:
      1. **Velocity MSE** — ``MSE(code_vel, velocity_target)``
      2. **Reconstruction MSE** — ``MSE(reconstruction, policy_obs)``
      3. **KL divergence** — ``β · KL(q(z|x) ‖ N(0, I))`` on the latent branch.

    Args:
        input_dim:  Dimension of the flattened observation history.
        output_dim: Dimension of the reconstructed observation (policy obs size).
        code_dim:   Total dimension of the latent code (``velocity_dim + latent_dim``).
        velocity_dim: Dimension of the velocity sub-code.
        encoder_hidden_dims: Hidden-layer sizes for the encoder MLP.
        encoder_latent_dim: Output size of the encoder MLP (before the Gaussian heads).
        decoder_hidden_dims: Hidden-layer sizes for the decoder MLP.
        activation: Activation function name (e.g. ``"elu"``, ``"relu"``).
    """
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        code_dim: int,
        velocity_dim: int,
        encoder_hidden_dims: list[int]|tuple[int, ...] = (128, ),
        encoder_latent_dim: int = 64,
        decoder_hidden_dims: list[int]|tuple[int, ...] = (128, 64),
        activation: str = "elu",
        ) -> None:
        super().__init__()

        self.code_dim = code_dim
        self.velocity_dim = velocity_dim
        self.latent_dim = code_dim - velocity_dim

        # h = MLP([128, 64], act=ELU)(o_hist)   # (B, 64)
        self.encoder = MLP(
            input_dim,
            encoder_latent_dim,
            list(encoder_hidden_dims),
            activation,
            last_activation=activation
        )

        # Gaussian heads
        # v_hat = Linear(64, 3)(h)  # (B, 3) 速度估计头
        self.mean_vel = nn.Linear(encoder_latent_dim, velocity_dim)
        self.logvar_vel = nn.Linear(encoder_latent_dim, velocity_dim)
        # mu_z    = Linear(64, 16)(h)  # (B, 16)  VAE均值头
        self.mean_latent = nn.Linear(encoder_latent_dim, self.latent_dim)
        # logvar_z= Linear(64, 16)(h) # (B, 16)  VAE对数方差头
        self.logvar_latent = nn.Linear(encoder_latent_dim, self.latent_dim)

        # Decoder: code → reconstructed observation
        self.decoder = MLP(code_dim, output_dim, list(decoder_hidden_dims), activation, last_activation=activation)

    @staticmethod
    def reparameterise(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.rand_like(std)

    def forward(self, obs_history: torch.Tensor) -> ContextVAEOutput:
        h = self.encoder(obs_history)

        vel_mean = self.mean_vel(h)
        vel_logvar = self.logvar_vel(h)
        latent_mean = self.mean_latent(h)
        latent_logvar = self.logvar_latent(h)

        vel_context = self.reparameterise(vel_mean, vel_logvar)
        latent_context = self.reparameterise(latent_mean, latent_logvar)
        code = torch.cat([vel_context, latent_context], dim=-1)

        reconstruction = self.decoder(code)

        return ContextVAEOutput(
            code=code,
            code_vel=vel_context,
            code_latent=latent_context,
            reconstruction=reconstruction,
            mean_vel=vel_mean,
            logvar_vel=vel_logvar,
            mean_latent=latent_mean,
            logvar_latent=latent_logvar,
        )
    
    def encode(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Encode only — returns the full latent code without decoding.

        Useful at inference time when the reconstruction is not needed.
        """
        h = self.encoder(obs_history)
        c_vel = self.reparameterise(self.mean_vel(h), self.logvar_vel(h))
        c_lat = self.reparameterise(self.mean_latent(h), self.logvar_latent(h))
        return torch.cat((c_vel, c_lat), dim=-1)

