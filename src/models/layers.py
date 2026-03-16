"""Shared building blocks: normalization, Fourier features, NTK dense, DGM gates."""
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict

from src.config import DTYPE


class Normalize(nn.Module):
    """Normalizes spatial and temporal coordinates to [-1, 1] range."""
    lx: float
    ly: float
    t_final: float
    x_min: float = 0.0
    y_min: float = 0.0

    @nn.compact
    def __call__(self, x):
        x_scaled = 2. * (x[..., 0] - self.x_min) / self.lx - 1.
        y_scaled = 2. * (x[..., 1] - self.y_min) / self.ly - 1.
        t_scaled = 2. * x[..., 2] / self.t_final - 1.
        return jnp.stack([x_scaled, y_scaled, t_scaled], axis=-1)


class FourierFeatures(nn.Module):
    """Projects inputs into higher-dimensional frequency space to overcome spectral bias."""
    output_dims: int
    scale: float

    @nn.compact
    def __call__(self, x):
        B = self.param('B', nn.initializers.normal(stddev=self.scale),
                        (x.shape[-1], self.output_dims // 2))
        x_proj = x @ B
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class NTKDense(nn.Module):
    """A Dense layer with NTK parameterization."""
    features: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param('kernel',
                            jax.nn.initializers.normal(stddev=1.0),
                            (x.shape[-1], self.features))
        bias = self.param('bias',
                          jax.nn.initializers.normal(stddev=1.0),
                          (self.features,))
        return jnp.dot(x, kernel) / jnp.sqrt(x.shape[-1]) + bias


class DGMLayer(nn.Module):
    """A single DGM layer inspired by LSTM."""
    num_units: int
    input_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, s_prev: jnp.ndarray) -> jnp.ndarray:
        Uz = nn.Dense(self.num_units, name='Uz', kernel_init=nn.initializers.glorot_uniform())
        Ug = nn.Dense(self.num_units, name='Ug', kernel_init=nn.initializers.glorot_uniform())
        Ur = nn.Dense(self.num_units, name='Ur', kernel_init=nn.initializers.glorot_uniform())
        Uh = nn.Dense(self.num_units, name='Uh', kernel_init=nn.initializers.glorot_uniform())

        Wz = nn.Dense(self.num_units, use_bias=False, name='Wz', kernel_init=nn.initializers.glorot_uniform())
        Wg = nn.Dense(self.num_units, use_bias=False, name='Wg', kernel_init=nn.initializers.glorot_uniform())
        Wr = nn.Dense(self.num_units, use_bias=False, name='Wr', kernel_init=nn.initializers.glorot_uniform())
        Wh = nn.Dense(self.num_units, use_bias=False, name='Wh', kernel_init=nn.initializers.glorot_uniform())

        Z = nn.tanh(Uz(x) + Wz(s_prev))
        G = nn.tanh(Ug(x) + Wg(s_prev))
        R = nn.tanh(Ur(x) + Wr(s_prev))
        H = nn.tanh(Uh(x) + Wh(s_prev * R))
        s_next = (1 - G) * H + Z * s_prev
        return s_next


def apply_output_scaling(x: jnp.ndarray, config: FrozenDict) -> jnp.ndarray:
    """Multiplies the raw network output by physical scale factors."""
    scales = config["model"].get("output_scales", None)
    if scales is not None:
        scale_tensor = jnp.array(scales, dtype=DTYPE)
        return x * scale_tensor
    return x
