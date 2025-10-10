# src/models.py
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Dict, Tuple
from src.config import config, DTYPE

# --- Parameters (remain the same) ---
WIDTH = config["model"]["width"]
DEPTH = config["model"]["depth"]
OUTPUT_DIM = config["model"]["output_dim"]
KERNEL_INIT = nn.initializers.glorot_uniform()
BIAS_INIT = nn.initializers.constant(config["model"]["bias_init"])
LX = config["domain"]["lx"]
LY = config["domain"]["ly"]
T_FINAL = config["domain"]["t_final"]

# --- Normalizer (remains the same) ---
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

# --- NEW: Fourier Feature Mapping Module ---
class FourierFeatures(nn.Module):
    output_dims: int  # The desired dimension of the feature vector
    scale: float = 10.0  # A hyperparameter controlling the frequency range

    @nn.compact
    def __call__(self, x):
        # Sample a random but fixed frequency matrix B
        # Ensure B is not re-initialized on every call
        B = self.param('B', nn.initializers.normal(stddev=self.scale), (x.shape[-1], self.output_dims // 2))
        
        # Calculate the Fourier features
        x_proj = x @ B
        features = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
        return features

# --- UPDATED: PINN Model ---
class PINN(nn.Module):
    """PINN with Fourier Feature Mapping."""
    # MLP parameters
    WIDTH: int = WIDTH
    DEPTH: int = DEPTH
    OUTPUT_DIM: int = OUTPUT_DIM
    # Normalization parameters
    lx: float = LX
    ly: float = LY
    t_final: float = T_FINAL
    # Initializers
    BIAS_INIT: Any = BIAS_INIT
    kernel_init: Any = KERNEL_INIT
    # Fourier Feature parameters
    FF_DIMS: int = 256 # Number of fourier features, a key hyperparameter

    def setup(self):
        self.normalizer = Normalize(self.lx, self.ly, self.t_final)
        self.fourier_features = FourierFeatures(output_dims=self.FF_DIMS)
        
        # Define the dense layers for the MLP part
        self.dense_layers = [nn.Dense(self.WIDTH, kernel_init=self.kernel_init, bias_init=self.BIAS_INIT) for _ in range(self.DEPTH)]
        self.output_layer = nn.Dense(self.OUTPUT_DIM, kernel_init=self.kernel_init, bias_init=self.BIAS_INIT)

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # 1. Normalize coordinates to [-1, 1]
        x_norm = self.normalizer(x)
        
        # 2. Create Fourier Features
        x_features = self.fourier_features(x_norm)
        
        # 3. Process features with the standard MLP
        for layer in self.dense_layers:
            x_features = nn.tanh(layer(x_features))
        
        return self.output_layer(x_features)

def init_model(key: jax.random.PRNGKey) -> Tuple[PINN, Dict[str, Any]]:
    """Initialize the PINN model and parameters."""
    model = PINN()
    # In `src/train.py` change `init_mlp` to `init_model`
    variables = model.init(key, jnp.zeros((1, 3), dtype=DTYPE))
    params = variables['params']
    return model, {'params': params}