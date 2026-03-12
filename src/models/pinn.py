"""Core PINN architectures: FourierPINN, MLP, DGMNetwork."""
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict

from src.models.layers import Normalize, FourierFeatures, DGMLayer, apply_output_scaling


class FourierPINN(nn.Module):
    """PINN with Fourier Feature Mapping."""
    config: FrozenDict

    def setup(self):
        model_cfg = self.config["model"]
        domain_cfg = self.config["domain"]

        self.normalizer = Normalize(
            lx=domain_cfg["lx"],
            ly=domain_cfg["ly"],
            t_final=domain_cfg["t_final"],
            x_min=domain_cfg.get("x_min", 0.0),
            y_min=domain_cfg.get("y_min", 0.0)
        )

        self.fourier_features = FourierFeatures(
            output_dims=model_cfg["ff_dims"],
            scale=model_cfg["fourier_scale"]
        )

        dense_layers = []
        for _ in range(model_cfg["depth"]):
            dense_layers.append(nn.Dense(
                model_cfg["width"],
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.constant(model_cfg["bias_init"])
            ))
        self.dense_layers = dense_layers

        self.output_layer = nn.Dense(
            model_cfg["output_dim"],
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.constant(model_cfg["bias_init"])
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x_norm = self.normalizer(x)
        x_features = self.fourier_features(x_norm)

        for layer in self.dense_layers:
            x_features = nn.tanh(layer(x_features))

        output = self.output_layer(x_features)
        return apply_output_scaling(output, self.config)

class MLP(nn.Module):
    """Standard Multi-Layer Perceptron."""
    config: FrozenDict

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        model_cfg = self.config["model"]
        domain_cfg = self.config["domain"]

        # Normalize input with offsets
        x = Normalize(
            lx=domain_cfg["lx"],
            ly=domain_cfg["ly"],
            t_final=domain_cfg["t_final"],
            x_min=domain_cfg.get("x_min", 0.0),
            y_min=domain_cfg.get("y_min", 0.0)
        )(x)

        # Hidden layers with tanh activation
        for _ in range(model_cfg["depth"]):
            x = nn.Dense(
                model_cfg["width"],
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.constant(model_cfg.get("bias_init", 0.0))
            )(x)
            x = nn.tanh(x)

        # Output layer
        x = nn.Dense(
            model_cfg["output_dim"],
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.constant(model_cfg.get("bias_init", 0.0)),
            name='output_layer'
            )(x)

        return x


class DGMNetwork(nn.Module):
    """Deep Galerkin Method Network."""
    config: FrozenDict

    @nn.compact
    def __call__(self, x_input: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        model_cfg = self.config["model"]
        domain_cfg = self.config["domain"]

        num_layers = model_cfg.get("depth", 3)
        num_units = model_cfg.get("width", 50)
        output_dim = model_cfg.get("output_dim", 3)
        input_dim = x_input.shape[-1]

        # Normalization layer with Offsets
        x_norm = Normalize(
            lx=domain_cfg["lx"],
            ly=domain_cfg["ly"],
            t_final=domain_cfg["t_final"],
            x_min=domain_cfg.get("x_min", 0.0),
            y_min=domain_cfg.get("y_min", 0.0)
        )(x_input)

        s_current = nn.tanh(nn.Dense(num_units, name='InitialDense', kernel_init=nn.initializers.glorot_uniform())(x_norm))

        for i in range(num_layers):
            s_current = DGMLayer(num_units=num_units, input_dim=input_dim, name=f'DGMLayer_{i}')(x_norm, s_current)

        output = nn.Dense(output_dim, name='output_layer', kernel_init=nn.initializers.glorot_uniform())(s_current)

        return apply_output_scaling(output, self.config)
