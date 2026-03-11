"""
Flax module definitions for the three PINN architectures.

These are reference implementations showing the canonical structure.
The production versions live in src/models.py — always check there for
the authoritative implementation.

Depends on: jax, flax.linen
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence


class MLP(nn.Module):
    """Standard multi-layer perceptron for PINN.

    Attributes
    ----------
    hidden_layers : int
        Number of hidden layers.
    hidden_units : int
        Units per hidden layer.
    output_dim : int
        Output dimension (3 for h, hu, hv).
    activation : str
        Activation function name ('tanh', 'relu', 'gelu').
    """
    hidden_layers: int = 6
    hidden_units: int = 128
    output_dim: int = 3
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x, y, t):
        act_fn = {"tanh": nn.tanh, "relu": nn.relu, "gelu": nn.gelu}[self.activation]
        z = jnp.stack([x, y, t], axis=-1)

        for _ in range(self.hidden_layers):
            z = nn.Dense(self.hidden_units)(z)
            z = act_fn(z)

        out = nn.Dense(self.output_dim)(z)
        return out[..., 0], out[..., 1], out[..., 2]  # h, hu, hv


class FourierPINN(nn.Module):
    """Fourier feature encoding + MLP to overcome spectral bias.

    Projects inputs into a higher-dimensional frequency space before
    passing through a standard MLP.

    Attributes
    ----------
    mapping_size : int
        Number of Fourier features (output dim = 2 * mapping_size).
    frequency_scale : float
        Scale of the random frequency matrix.
    hidden_layers : int
    hidden_units : int
    output_dim : int
    """
    mapping_size: int = 128
    frequency_scale: float = 1.0
    hidden_layers: int = 6
    hidden_units: int = 128
    output_dim: int = 3

    @nn.compact
    def __call__(self, x, y, t):
        z = jnp.stack([x, y, t], axis=-1)

        # Fourier feature mapping: z -> [cos(Bz), sin(Bz)]
        # B is a fixed random matrix (not trained)
        B = self.param(
            "fourier_B",
            nn.initializers.normal(stddev=self.frequency_scale),
            (z.shape[-1], self.mapping_size),
        )
        projected = z @ B  # (..., mapping_size)
        z = jnp.concatenate([jnp.cos(projected), jnp.sin(projected)], axis=-1)

        # Standard MLP on top
        for _ in range(self.hidden_layers):
            z = nn.Dense(self.hidden_units)(z)
            z = nn.tanh(z)

        out = nn.Dense(self.output_dim)(z)
        return out[..., 0], out[..., 1], out[..., 2]


class DGMNetwork(nn.Module):
    """Deep Galerkin Method network with highway-style gating.

    Injects raw coordinates into every hidden layer via gated sub-networks,
    allowing the network to maintain awareness of the input throughout.

    Attributes
    ----------
    hidden_layers : int
    hidden_units : int
    output_dim : int
    """
    hidden_layers: int = 6
    hidden_units: int = 128
    output_dim: int = 3

    @nn.compact
    def __call__(self, x, y, t):
        raw = jnp.stack([x, y, t], axis=-1)

        # Initial projection
        S = nn.Dense(self.hidden_units)(raw)
        S = nn.tanh(S)

        for _ in range(self.hidden_layers):
            # Gate sub-networks (LSTM-like structure)
            Z = nn.sigmoid(nn.Dense(self.hidden_units)(raw) + nn.Dense(self.hidden_units)(S))
            G = nn.sigmoid(nn.Dense(self.hidden_units)(raw) + nn.Dense(self.hidden_units)(S))
            R = nn.sigmoid(nn.Dense(self.hidden_units)(raw) + nn.Dense(self.hidden_units)(S))
            H = nn.tanh(nn.Dense(self.hidden_units)(raw) + nn.Dense(self.hidden_units)(S * R))

            S = (1 - G) * H + Z * S

        out = nn.Dense(self.output_dim)(S)
        return out[..., 0], out[..., 1], out[..., 2]
