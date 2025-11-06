# src/models.py
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict
from typing import Any, Dict, Tuple, Sequence
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

class MLP(nn.Module):
    """Standard Multi-Layer Perceptron."""
    config: FrozenDict

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        model_cfg = self.config["model"]
        domain_cfg = self.config["domain"]

        # Normalize input
        x = Normalize(lx=domain_cfg["lx"], ly=domain_cfg["ly"], t_final=domain_cfg["t_final"])(x)

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
            bias_init=nn.initializers.constant(model_cfg.get("bias_init", 0.0))
        )(x)

        return x
    
    
# --- Add the DGMLayer Class ---
class DGMLayer(nn.Module):
    """A single DGM layer inspired by LSTM."""
    num_units: int # Number of units (M in the paper)
    input_dim: int # Dimension of the original input x = (t, x_1, ..., x_d)

    @nn.compact
    def __call__(self, x: jnp.ndarray, s_prev: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the DGM layer transformation.

        Args:
            x: The original network input (batch, input_dim).
            s_prev: The state output from the previous layer (batch, num_units).

        Returns:
            s_next: The state output for the next layer (batch, num_units).
        """
        # Define dense layers for each gate/component
        # Note: kernel_init can be customized, glorot_uniform is a common default
        Uz = nn.Dense(self.num_units, name='Uz', kernel_init=nn.initializers.glorot_uniform())
        Ug = nn.Dense(self.num_units, name='Ug', kernel_init=nn.initializers.glorot_uniform())
        Ur = nn.Dense(self.num_units, name='Ur', kernel_init=nn.initializers.glorot_uniform())
        Uh = nn.Dense(self.num_units, name='Uh', kernel_init=nn.initializers.glorot_uniform())

        Wz = nn.Dense(self.num_units, use_bias=False, name='Wz', kernel_init=nn.initializers.glorot_uniform()) # Bias included in Uz
        Wg = nn.Dense(self.num_units, use_bias=False, name='Wg', kernel_init=nn.initializers.glorot_uniform()) # Bias included in Ug
        Wr = nn.Dense(self.num_units, use_bias=False, name='Wr', kernel_init=nn.initializers.glorot_uniform()) # Bias included in Ur
        Wh = nn.Dense(self.num_units, use_bias=False, name='Wh', kernel_init=nn.initializers.glorot_uniform()) # Bias included in Uh

        # Calculate gates Z, G, R (Equation 4.2 in PDF)
        # Using tanh as the activation function sigma, as mentioned effective in paper
        Z = nn.tanh(Uz(x) + Wz(s_prev))
        G = nn.tanh(Ug(x) + Wg(s_prev))
        R = nn.tanh(Ur(x) + Wr(s_prev))

        # Calculate candidate state H (Equation 4.2 in PDF)
        H = nn.tanh(Uh(x) + Wh(s_prev * R)) # Element-wise product S * R

        # Calculate next state S (Equation 4.2 in PDF)
        s_next = (1 - G) * H + Z * s_prev # Element-wise products

        return s_next

# --- Add the DGMNetwork Class ---
class DGMNetwork(nn.Module):
    """Deep Galerkin Method Network."""
    config: FrozenDict # Configuration dictionary

    @nn.compact
    def __call__(self, x_input: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Applies the full DGM network transformation.

        Args:
            x_input: Input tensor (batch, input_dim = d+1 for space+time).
                     Assumes input shape (batch, 3) for (x, y, t).
            train: Boolean indicating if the model is in training mode (unused here).

        Returns:
            Output tensor (batch, output_dim).
        """
        model_cfg = self.config["model"]
        domain_cfg = self.config["domain"]

        # Configurable parameters (add these to your YAML)
        num_layers = model_cfg.get("depth", 3) # L in the paper (number of hidden DGM layers)
        num_units = model_cfg.get("width", 50) # M in the paper
        output_dim = model_cfg.get("output_dim", 3) # Your output dim (h, hu, hv)
        input_dim = x_input.shape[-1] # Should be 3 for (x, y, t)

        # Normalization layer (similar to your other models)
        x_norm = Normalize(lx=domain_cfg["lx"], ly=domain_cfg["ly"], t_final=domain_cfg["t_final"])(x_input)

        # Initial layer S1 (Equation 4.2 in PDF)
        s_current = nn.tanh(nn.Dense(num_units, name='InitialDense', kernel_init=nn.initializers.glorot_uniform())(x_norm))

        # Stacked DGM layers (L layers)
        for i in range(num_layers):
            s_current = DGMLayer(num_units=num_units, input_dim=input_dim, name=f'DGMLayer_{i}')(x_norm, s_current)
            # Note: The paper passes the original input 'x' to each layer.

        # Final output layer (Linear transformation, Equation 4.2 in PDF)
        output = nn.Dense(output_dim, name='OutputDense', kernel_init=nn.initializers.glorot_uniform())(s_current)

        return output


def init_model(model_class: nn.Module, key: jax.random.PRNGKey, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    """Initialize the PINN model and parameters."""
    model = model_class(config=config)
    variables = model.init(key, jnp.zeros((1, 3), dtype=DTYPE))
    return model, {'params': variables['params']}