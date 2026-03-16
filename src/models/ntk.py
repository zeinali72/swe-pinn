"""NTK-parameterized architectures."""
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict

from src.models.layers import Normalize, FourierFeatures, NTKDense


class NTK_MLP(nn.Module):
    """Multi-Layer Perceptron using NTK Parameterization."""
    config: FrozenDict

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        model_cfg = self.config["model"]
        domain_cfg = self.config["domain"]

        x = Normalize(
            lx=domain_cfg["lx"], ly=domain_cfg["ly"], t_final=domain_cfg["t_final"],
            x_min=domain_cfg.get("x_min", 0.0), y_min=domain_cfg.get("y_min", 0.0)
        )(x)

        for _ in range(model_cfg["depth"]):
            x = NTKDense(features=model_cfg["width"])(x)
            x = nn.tanh(x)

        return NTKDense(features=model_cfg["output_dim"])(x)


class FourierNTK_MLP(nn.Module):
    """PINN combining Fourier Feature Mapping and NTK Parameterization."""
    config: FrozenDict

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        model_cfg = self.config["model"]
        domain_cfg = self.config["domain"]

        x = Normalize(
            lx=domain_cfg["lx"], ly=domain_cfg["ly"], t_final=domain_cfg["t_final"],
            x_min=domain_cfg.get("x_min", 0.0), y_min=domain_cfg.get("y_min", 0.0)
        )(x)

        x = FourierFeatures(output_dims=model_cfg["ff_dims"], scale=model_cfg["fourier_scale"])(x)

        for _ in range(model_cfg["depth"]):
            x = NTKDense(features=model_cfg["width"])(x)
            x = nn.tanh(x)

        return NTKDense(features=model_cfg["output_dim"])(x)
