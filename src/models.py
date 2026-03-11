# src/models.py
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict
from typing import Any, Dict, Tuple, Sequence
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
        # Correctly shift to [0, lx] first, then normalize to [-1, 1]
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
        B = self.param('B', nn.initializers.normal(stddev=self.scale), (x.shape[-1], self.output_dims // 2))
        x_proj = x @ B
        features = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
        return features
    
    # --- HELPER: Output Scaling ---
def apply_output_scaling(x: jnp.ndarray, config: FrozenDict) -> jnp.ndarray:
    """Multiplies the raw network output by physical scale factors."""
    # Check if 'output_scales' exists in the model config
    scales = config["model"].get("output_scales", None)
    if scales is not None:
        # scales should be a list/tuple like [50.0, 10.0, 10.0] for [h, hu, hv]
        scale_tensor = jnp.array(scales, dtype=DTYPE)
        return x * scale_tensor
    return x

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
    
def init_model(model_class: nn.Module, key: jax.random.PRNGKey, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    """Initialize the PINN model and parameters."""
    model = model_class(config=config)
    variables = model.init(key, jnp.zeros((1, 3), dtype=DTYPE))
    return model, {'params': variables['params']}

class DeepONet(nn.Module):
    """DeepONet architecture."""
    config: FrozenDict

    @nn.compact
    def __call__(self, x_branch: jnp.ndarray, x_trunk: jnp.ndarray, train: bool = True) -> jnp.ndarray:
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

        # --- Branch Network ---
        b = x_branch
        for _ in range(model_cfg["branch_depth"]):
            b = nn.Dense(model_cfg["branch_width"], kernel_init=kernel_init, bias_init=bias_init)(b)
            b = nn.tanh(b)
        branch_out = nn.Dense(p_dim * output_dim, name="branch_output", kernel_init=kernel_init, bias_init=bias_init)(b)
        branch_out = branch_out.reshape(-1, output_dim, p_dim)

        # --- Trunk Network ---
        # Normalize with offsets
        t = Normalize(
            lx=domain_cfg["lx"], 
            ly=domain_cfg["ly"], 
            t_final=domain_cfg["t_final"],
            x_min=domain_cfg.get("x_min", 0.0),
            y_min=domain_cfg.get("y_min", 0.0)
        )(x_trunk)

        for _ in range(model_cfg["trunk_depth"]):
            t = nn.Dense(model_cfg["trunk_width"], kernel_init=kernel_init, bias_init=bias_init)(t)
            t = nn.tanh(t)
        trunk_out = nn.Dense(p_dim, name="trunk_output", kernel_init=kernel_init, bias_init=bias_init)(t)

        bias_t = self.param('bias_t', nn.initializers.zeros, (output_dim,))

        output = jnp.sum(branch_out * trunk_out[:, None, :], axis=-1) + bias_t
        
        if is_unbatched:
            output = output[0]
            
        return output

class FourierDeepONet(nn.Module):
    """DeepONet with Fourier-enhanced trunk network."""
    config: FrozenDict

    @nn.compact
    def __call__(self, x_branch: jnp.ndarray, x_trunk: jnp.ndarray, train: bool = True) -> jnp.ndarray:
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

        # --- Branch Network ---
        b = x_branch
        for _ in range(model_cfg["branch_depth"]):
            b = nn.Dense(model_cfg["branch_width"], kernel_init=kernel_init, bias_init=bias_init)(b)
            b = nn.tanh(b)
        branch_out = nn.Dense(p_dim * output_dim, name="branch_output", kernel_init=kernel_init, bias_init=bias_init)(b)
        branch_out = branch_out.reshape(-1, output_dim, p_dim)

        # --- Trunk Network ---
        # Normalize with offsets
        t = Normalize(
            lx=domain_cfg["lx"], 
            ly=domain_cfg["ly"], 
            t_final=domain_cfg["t_final"],
            x_min=domain_cfg.get("x_min", 0.0),
            y_min=domain_cfg.get("y_min", 0.0)
        )(x_trunk)
        
        fourier_dims = model_cfg.get("trunk_fourier_dims", 256)
        fourier_scale = model_cfg.get("trunk_fourier_scale", 1.0)
        t = FourierFeatures(output_dims=fourier_dims, scale=fourier_scale)(t)
        
        for _ in range(model_cfg["trunk_depth"]):
            t = nn.Dense(model_cfg["trunk_width"], kernel_init=kernel_init, bias_init=bias_init)(t)
            t = nn.tanh(t)
        trunk_out = nn.Dense(p_dim, name="trunk_output", kernel_init=kernel_init, bias_init=bias_init)(t)

        bias_t = self.param('bias_t', nn.initializers.zeros, (output_dim,))

        output = jnp.sum(branch_out * trunk_out[:, None, :], axis=-1) + bias_t
        
        if is_unbatched:
            output = output[0]
            
        return output

def init_deeponet_model(model_class: nn.Module, key: jax.random.PRNGKey, config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    """Initialize a DeepONet model with branch and trunk networks."""
    model = model_class(config=config)
    param_names = tuple(config["physics"]["param_bounds"].keys())
    n_params = len(param_names)
    if n_params == 0:
        raise ValueError("Config 'physics.param_bounds' is empty. DeepONet branch net has no inputs.")
        
    dummy_input_branch = jnp.zeros((1, n_params), dtype=DTYPE)
    dummy_input_trunk = jnp.zeros((1, 3), dtype=DTYPE) 
    
    print(f"Initializing DeepONet with branch input shape: {dummy_input_branch.shape} (params: {param_names})")
    print(f"Initializing DeepONet with trunk input shape: {dummy_input_trunk.shape} (coords: x, y, t)")
    
    variables = model.init(key, dummy_input_branch, dummy_input_trunk)
    return model, {'params': variables['params']}

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

class NTK_MLP(nn.Module):
    """Multi-Layer Perceptron using NTK Parameterization."""
    config: FrozenDict

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        model_cfg = self.config["model"]
        domain_cfg = self.config["domain"]

        # Normalize with offsets
        x = Normalize(
            lx=domain_cfg["lx"], 
            ly=domain_cfg["ly"], 
            t_final=domain_cfg["t_final"],
            x_min=domain_cfg.get("x_min", 0.0),
            y_min=domain_cfg.get("y_min", 0.0)
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

        # Normalize with offsets
        x = Normalize(
            lx=domain_cfg["lx"], 
            ly=domain_cfg["ly"], 
            t_final=domain_cfg["t_final"],
            x_min=domain_cfg.get("x_min", 0.0),
            y_min=domain_cfg.get("y_min", 0.0)
        )(x)

        x = FourierFeatures(output_dims=model_cfg["ff_dims"], scale=model_cfg["fourier_scale"])(x)

        for _ in range(model_cfg["depth"]):
            x = NTKDense(features=model_cfg["width"])(x)
            x = nn.tanh(x)

        return NTKDense(features=model_cfg["output_dim"])(x)