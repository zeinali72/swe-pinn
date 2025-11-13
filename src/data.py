# src/data.py
import jax
import jax.numpy as jnp
from jax import random
from src.config import DTYPE
from typing import Dict, Tuple, Callable # <-- Added imports

def sample_domain(key: jax.random.PRNGKey, 
                  n_total: int, 
                  x_range: Tuple[float, float], 
                  y_range: Tuple[float, float], 
                  t_range: Tuple[float, float]) -> jnp.ndarray:
    """Samples n_total points uniformly in a 3D domain."""
    
    # Handle zero-point requests cleanly
    if n_total == 0:
        return jnp.empty((0, 3), dtype=DTYPE)
        
    key_x, key_y, key_t = jax.random.split(key, 3)
    
    # Handle singular dimensions (like t_start=t_end for IC)
    if x_range[0] == x_range[1]:
        x_coords = jnp.full((n_total, 1), x_range[0], dtype=DTYPE)
    else:
        x_coords = random.uniform(key_x, (n_total, 1), minval=x_range[0], maxval=x_range[1], dtype=DTYPE)

    if y_range[0] == y_range[1]:
        y_coords = jnp.full((n_total, 1), y_range[0], dtype=DTYPE)
    else:
        y_coords = random.uniform(key_y, (n_total, 1), minval=y_range[0], maxval=y_range[1], dtype=DTYPE)
        
    if t_range[0] == t_range[1]:
        t_coords = jnp.full((n_total, 1), t_range[0], dtype=DTYPE)
    else:
        t_coords = random.uniform(key_t, (n_total, 1), minval=t_range[0], maxval=t_range[1], dtype=DTYPE)
    
    return jnp.hstack([x_coords, y_coords, t_coords])

def get_batches(key: jax.random.PRNGKey, data: jnp.ndarray, batch_size: int) -> list:
    """Shuffle and split data into batches."""
    data = jax.random.permutation(key, data, axis=0)
    return [data[i:i + batch_size].astype(DTYPE) for i in range(0, data.shape[0], batch_size)]

def sample_parameters(key: jax.random.PRNGKey, param_bounds: dict[str, tuple[float, float]], n_samples: int) -> tuple[jnp.ndarray, tuple[str, ...]]:
    """
    Samples parameters from given bounds using a dictionary.
    """
    if not param_bounds:
        raise ValueError("param_bounds must contain at least one parameter.")
    
    names = tuple(param_bounds.keys())
    subkeys = random.split(key, len(names))
    columns = []
    
    print(f"[Data] Sampling {n_samples} sets for parameters: {names}")
    
    for (lower, upper), subkey in zip(param_bounds.values(), subkeys):
        if lower == upper:
            columns.append(jnp.full((n_samples, 1), lower, dtype=DTYPE))
        else:
            columns.append(random.uniform(subkey, (n_samples, 1), minval=lower, maxval=upper, dtype=DTYPE))
            
    return jnp.hstack(columns), names

# <<<--- NEW FUNCTION (Moved from train_deeponet.py) --->>>
def create_operator_dataset(
    key: jax.random.PRNGKey, 
    config: FrozenDict, 
    n_funcs: int, 
    n_points_per_func: int,
    solver: Callable,
    param_names: Tuple[str, ...]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Creates a paired dataset of (parameters, coordinates, true_solution_h)
    for a single epoch or validation.
    """
    key_params, key_points = random.split(key)
    
    param_bounds = config["physics"]["param_bounds"]
    domain_cfg = config["domain"]
    
    # 1. Sample Branch Inputs (Parameters) - (n_funcs, n_params)
    branch_inputs, _ = sample_parameters(key_params, param_bounds, n_funcs)
    
    # 2. Sample Trunk Inputs (Coordinates) - (n_funcs * n_points, 3)
    trunk_inputs_flat = sample_points(
        0., domain_cfg["lx"], 0., domain_cfg["ly"], 0., domain_cfg["t_final"],
        n_funcs * n_points_per_func, 1, 1, key_points
    )
    
    # 3. Create paired dataset
    # branch_inputs becomes (n_funcs * n_points, n_params)
    branch_inputs_paired = jnp.repeat(branch_inputs, n_points_per_func, axis=0)
    trunk_inputs_paired = trunk_inputs_flat # (n_funcs * n_points, 3)
    
    # 4. Compute True Outputs (Analytical Solution for h)
    param_map = {name: i for i, name in enumerate(param_names)}
    n_manning_idx = param_map.get('n_manning')
    u_const_idx = param_map.get('u_const')
    
    n_manning_default = config["physics"]["n_manning"]
    u_const_default = config["physics"]["u_const"]

    n_manning = branch_inputs_paired[..., n_manning_idx] if n_manning_idx is not None else jnp.full(branch_inputs_paired.shape[0], n_manning_default, dtype=DTYPE)
    u_const = branch_inputs_paired[..., u_const_idx] if u_const_idx is not None else jnp.full(branch_inputs_paired.shape[0], u_const_default, dtype=DTYPE)
    
    x_coords = trunk_inputs_paired[..., 0]
    t_coords = trunk_inputs_paired[..., 2]
    
    true_outputs_h = solver(x_coords, t_coords, n_manning, u_const)
    
    return branch_inputs_paired, trunk_inputs_paired, true_outputs_h[..., None] # Shape (N, 1)