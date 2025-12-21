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
            ,name='output_layer'
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
        output = nn.Dense(output_dim, name='output_layer', kernel_init=nn.initializers.glorot_uniform())(s_current)

        return output

def init_model(model_class: nn.Module, key: jax.random.PRNGKey, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    """Initialize the PINN model and parameters."""
    model = model_class(config=config)
    variables = model.init(key, jnp.zeros((1, 3), dtype=DTYPE))
    return model, {'params': variables['params']}

class DeepONet(nn.Module):
    """
    DeepONet architecture for mapping parameters to solutions.
    - Branch net processes parameters (e.g., n_manning, u_const).
    - Trunk net processes coordinates (x, y, t).
    - Output dimension must match PINN models (e.g., 3 for [h, hu, hv])
      to be compatible with the physics loss functions.
    """
    config: FrozenDict

    @nn.compact
    def __call__(self, x_branch: jnp.ndarray, x_trunk: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Forward pass for the DeepONet.

        Args:
            x_branch: Branch inputs (physical parameters), shape (batch, n_params)
            x_trunk: Trunk inputs (coordinates x, y, t), shape (batch, 3)
            train: (Unused) Kept for compatibility with some loss function calls.

        Returns:
            Predicted output U(x,y,t), shape (batch, output_dim)
        """
        model_cfg = self.config["model"]
        domain_cfg = self.config["domain"]
        
        # Latent dimension
        p_dim = model_cfg["latent_dim"]
        output_dim = model_cfg["output_dim"] # e.g., 3 for [h, hu, hv]
        kernel_init = nn.initializers.glorot_uniform()
        bias_init = nn.initializers.constant(model_cfg.get("bias_init", 0.0))

        # --- Branch Network (processes parameters) ---
        b = x_branch
        for _ in range(model_cfg["branch_depth"]):
            b = nn.Dense(model_cfg["branch_width"], kernel_init=kernel_init, bias_init=bias_init)(b)
            b = nn.tanh(b)
        # Branch output, shape (batch, p_dim * output_dim)
        # We need p_dim outputs for *each* output variable (h, hu, hv)
        branch_out = nn.Dense(p_dim * output_dim, name="branch_output", kernel_init=kernel_init, bias_init=bias_init)(b)
        # Reshape to (batch, output_dim, p_dim)
        branch_out = branch_out.reshape(-1, output_dim, p_dim)

        # --- Trunk Network (processes coordinates [x, y, t]) ---
        # Normalize coordinates first
        t = Normalize(lx=domain_cfg["lx"], ly=domain_cfg["ly"], t_final=domain_cfg["t_final"])(x_trunk)
        for _ in range(model_cfg["trunk_depth"]):
            t = nn.Dense(model_cfg["trunk_width"], kernel_init=kernel_init, bias_init=bias_init)(t)
            t = nn.tanh(t)
        # Trunk output, shape (batch, p_dim)
        trunk_out = nn.Dense(p_dim, name="trunk_output", kernel_init=kernel_init, bias_init=bias_init)(t)

        # --- Add bias term (as in the original DeepONet paper) ---
        # One bias per output dimension
        bias_t = self.param('bias_t', nn.initializers.zeros, (output_dim,))

        # --- Combine outputs ---
        # (batch, output_dim, p_dim) * (batch, 1, p_dim) -> (batch, output_dim, p_dim)
        # Then sum over the latent dimension (axis=-1)
        # Resulting shape: (batch, output_dim)
        output = jnp.sum(branch_out * trunk_out[:, None, :], axis=-1)
        
        # Add trunk bias
        output = output + bias_t
        
        # This is the "shared layer" for GradNorm. We rename the trunk_output.
        # This is a trick: GradNorm will compute gradients w.r.t. the trunk_output layer,
        # which is a good representation of the shared computation.
        s = nn.Dense(output_dim, name='output_layer', kernel_init=kernel_init, bias_init=bias_init)(trunk_out) 
        
        # Final combination
        # (batch, output_dim, p_dim) * (batch, 1, p_dim) -> sum -> (batch, output_dim)
        output = jnp.sum(branch_out * trunk_out[:, None, :], axis=-1) + bias_t
        return output


def init_deeponet_model(model_class: nn.Module, key: jax.random.PRNGKey, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Initialize the DeepONet model and parameters.
    This is separate from init_model to avoid changing the original function.
    """
    model = model_class(config=config)
    
    # DeepONet needs two dummy inputs for initialization
    param_names = tuple(config["physics"]["param_bounds"].keys())
    n_params = len(param_names)
    if n_params == 0:
        raise ValueError("Config 'physics.param_bounds' is empty. DeepONet branch net has no inputs.")
        
    dummy_input_branch = jnp.zeros((1, n_params), dtype=DTYPE)
    dummy_input_trunk = jnp.zeros((1, 3), dtype=DTYPE) # (x, y, t)
    
    print(f"Initializing DeepONet with branch input shape: {dummy_input_branch.shape} (params: {param_names})")
    print(f"Initializing DeepONet with trunk input shape: {dummy_input_trunk.shape} (coords: x, y, t)")
    
    variables = model.init(key, dummy_input_branch, dummy_input_trunk)
    return model, {'params': variables['params']}



class NTKDense(nn.Module):
    features: int
    def setup(self):
        # Weights initialized to N(0, 1) as per Eq 2.1 & 2.2
        self.kernel = self.param('kernel', nn.initializers.normal(stddev=1.0), 
                                (self.input_shape[-1], self.features))
        self.bias = self.param('bias', nn.initializers.normal(stddev=1.0), (self.features,))

    def __call__(self, x):
        # Scale output by 1/sqrt(width)
        return (x @ self.kernel) / jnp.sqrt(self.features) + self.bias