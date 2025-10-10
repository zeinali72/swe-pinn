# src/models.py
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict
from typing import Any, Dict, Tuple
from src.config import DTYPE

class Normalize(nn.Module):
    lx: float
    ly: float
    t_final: float
    @nn.compact
    def __call__(self, x):
        x_scaled = 2. * x[..., 0] / self.lx - 1.
        y_scaled = 2. * x[..., 1] / self.ly - 1.
        t_scaled = 2. * x[..., 2] / self.t_final - 1.
        return jnp.stack([x_scaled, y_scaled, t_scaled], axis=-1)

class FourierFeatures(nn.Module):
    output_dims: int
    scale: float

    @nn.compact
    def __call__(self, x):
        B = self.param('B', nn.initializers.normal(stddev=self.scale), (x.shape[-1], self.output_dims // 2))
        x_proj = x @ B
        features = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
        return features

class FourierPINN(nn.Module):
    """PINN with Fourier Feature Mapping."""
    config: FrozenDict

    def setup(self):
        model_cfg = self.config["model"]
        domain_cfg = self.config["domain"]

        self.normalizer = Normalize(lx=domain_cfg["lx"], ly=domain_cfg["ly"], t_final=domain_cfg["t_final"])

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

        return self.output_layer(x_features)

class SirenLayer(nn.Module):
    """A single layer of a SIREN model."""
    features: int
    config: FrozenDict
    is_first: bool = False

    @nn.compact
    def __call__(self, x):
        model_cfg = self.config["model"]
        input_dim = x.shape[-1]

        # Correct initialization for SIREN layers
        w_std = (jnp.sqrt(6.0 / input_dim) / model_cfg["w0"]) if self.is_first else (jnp.sqrt(6.0 / input_dim) * model_cfg["w_init_factor"])
        w_init = jax.nn.initializers.uniform(scale=w_std)

        w = self.param('kernel', w_init, (input_dim, self.features))
        b = self.param('bias', jax.nn.initializers.zeros, (self.features,))
        # FIX: The lambda function now accepts two arguments (key, shape)
        w0_init_fn = lambda key, shape, dtype=jnp.float32: jnp.full(shape, model_cfg["w0"], dtype=dtype)
        w0 = self.param('w0', w0_init_fn, (1,)) if self.is_first else model_cfg["w0"]


        y = x @ w + b
        return jnp.sin(w0 * y)

class SIREN(nn.Module):
    """SIREN model with trainable w0."""
    config: FrozenDict

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        model_cfg = self.config["model"]
        domain_cfg = self.config["domain"]

        x = Normalize(lx=domain_cfg["lx"], ly=domain_cfg["ly"], t_final=domain_cfg["t_final"])(x)

        # Input layer
        x = SirenLayer(features=model_cfg["width"], is_first=True, config=self.config)(x)

        # Hidden layers
        for _ in range(model_cfg["depth"] - 1):
            x = SirenLayer(features=model_cfg["width"], is_first=False, config=self.config)(x)

        # Output layer
        # The output layer of SIREN is typically linear
        output_layer = nn.Dense(features=model_cfg["output_dim"])
        return output_layer(x)


def init_model(model_class: nn.Module, key: jax.random.PRNGKey, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    """Initialize the PINN model and parameters."""
    model = model_class(config=config)
    variables = model.init(key, jnp.zeros((1, 3), dtype=DTYPE))
    return model, {'params': variables['params']}