"""DeepONet operator network architectures."""
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict

from src.models.layers import Normalize, FourierFeatures


class DeepONet(nn.Module):
    """DeepONet architecture."""
    config: FrozenDict

    @nn.compact
    def __call__(self, x_branch: jnp.ndarray, x_trunk: jnp.ndarray,
                 train: bool = True) -> jnp.ndarray:
        is_unbatched = (x_trunk.ndim == 1)
        if is_unbatched:
            x_branch = x_branch[None, ...]
            x_trunk = x_trunk[None, ...]

        model_cfg = self.config["model"]
        domain_cfg = self.config["domain"]
        p_dim = model_cfg["latent_dim"]
        output_dim = model_cfg["output_dim"]
        kernel_init = nn.initializers.glorot_uniform()
        bias_init = nn.initializers.constant(model_cfg.get("bias_init", 0.0))

        # Branch Network
        b = x_branch
        for _ in range(model_cfg["branch_depth"]):
            b = nn.Dense(model_cfg["branch_width"], kernel_init=kernel_init, bias_init=bias_init)(b)
            b = nn.tanh(b)
        branch_out = nn.Dense(p_dim * output_dim, name="branch_output",
                              kernel_init=kernel_init, bias_init=bias_init)(b)
        branch_out = branch_out.reshape(-1, output_dim, p_dim)

        # Trunk Network
        t = Normalize(
            lx=domain_cfg["lx"], ly=domain_cfg["ly"], t_final=domain_cfg["t_final"],
            x_min=domain_cfg.get("x_min", 0.0), y_min=domain_cfg.get("y_min", 0.0)
        )(x_trunk)
        for _ in range(model_cfg["trunk_depth"]):
            t = nn.Dense(model_cfg["trunk_width"], kernel_init=kernel_init, bias_init=bias_init)(t)
            t = nn.tanh(t)
        trunk_out = nn.Dense(p_dim, name="trunk_output",
                             kernel_init=kernel_init, bias_init=bias_init)(t)

        bias_t = self.param('bias_t', nn.initializers.zeros, (output_dim,))
        output = jnp.sum(branch_out * trunk_out[:, None, :], axis=-1) + bias_t

        if is_unbatched:
            output = output[0]
        return output


class FourierDeepONet(nn.Module):
    """DeepONet with Fourier-enhanced trunk network."""
    config: FrozenDict

    @nn.compact
    def __call__(self, x_branch: jnp.ndarray, x_trunk: jnp.ndarray,
                 train: bool = True) -> jnp.ndarray:
        is_unbatched = (x_trunk.ndim == 1)
        if is_unbatched:
            x_branch = x_branch[None, ...]
            x_trunk = x_trunk[None, ...]

        model_cfg = self.config["model"]
        domain_cfg = self.config["domain"]
        p_dim = model_cfg["latent_dim"]
        output_dim = model_cfg["output_dim"]
        kernel_init = nn.initializers.glorot_uniform()
        bias_init = nn.initializers.constant(model_cfg.get("bias_init", 0.0))

        # Branch Network
        b = x_branch
        for _ in range(model_cfg["branch_depth"]):
            b = nn.Dense(model_cfg["branch_width"], kernel_init=kernel_init, bias_init=bias_init)(b)
            b = nn.tanh(b)
        branch_out = nn.Dense(p_dim * output_dim, name="branch_output",
                              kernel_init=kernel_init, bias_init=bias_init)(b)
        branch_out = branch_out.reshape(-1, output_dim, p_dim)

        # Trunk Network with Fourier features
        t = Normalize(
            lx=domain_cfg["lx"], ly=domain_cfg["ly"], t_final=domain_cfg["t_final"],
            x_min=domain_cfg.get("x_min", 0.0), y_min=domain_cfg.get("y_min", 0.0)
        )(x_trunk)
        fourier_dims = model_cfg.get("trunk_fourier_dims", 256)
        fourier_scale = model_cfg.get("trunk_fourier_scale", 1.0)
        t = FourierFeatures(output_dims=fourier_dims, scale=fourier_scale)(t)
        for _ in range(model_cfg["trunk_depth"]):
            t = nn.Dense(model_cfg["trunk_width"], kernel_init=kernel_init, bias_init=bias_init)(t)
            t = nn.tanh(t)
        trunk_out = nn.Dense(p_dim, name="trunk_output",
                             kernel_init=kernel_init, bias_init=bias_init)(t)

        bias_t = self.param('bias_t', nn.initializers.zeros, (output_dim,))
        output = jnp.sum(branch_out * trunk_out[:, None, :], axis=-1) + bias_t

        if is_unbatched:
            output = output[0]
        return output
