"""Model initialization helpers."""
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Dict, Tuple

from src.config import DTYPE


def init_model(model_class: nn.Module, key: jax.random.PRNGKey,
               config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    """Initialize a PINN model and return (model, params)."""
    model = model_class(config=config)
    variables = model.init(key, jnp.zeros((1, 3), dtype=DTYPE))
    return model, {'params': variables['params']}


def init_deeponet_model(model_class: nn.Module, key: jax.random.PRNGKey,
                        config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
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
