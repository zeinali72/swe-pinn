# src/data.py
import jax
import jax.numpy as jnp
from jax import random
from src.config import DTYPE

def sample_points(x_start: float, x_end: float, y_start: float, y_end: float,
                  t_start: float, t_end: float, nx: int, ny: int, nt: int, key) -> jnp.ndarray:
    """Sample points uniformly in a 3D domain, handling degenerate (singular) dimensions."""
    n_total = nx * ny * nt
    key_x, key_y, key_t = jax.random.split(key, 3)

    if x_start == x_end:
        x_coords = jnp.full((n_total, 1), x_start, dtype=DTYPE)
    else:
        x_coords = random.uniform(key_x, (n_total, 1), minval=x_start, maxval=x_end, dtype=DTYPE)

    if y_start == y_end:
        y_coords = jnp.full((n_total, 1), y_start, dtype=DTYPE)
    else:
        y_coords = random.uniform(key_y, (n_total, 1), minval=y_start, maxval=y_end, dtype=DTYPE)
        
    if t_start == t_end:
        t_coords = jnp.full((n_total, 1), t_start, dtype=DTYPE)
    else:
        t_coords = random.uniform(key_t, (n_total, 1), minval=t_start, maxval=t_end, dtype=DTYPE)
    
    return jnp.hstack([x_coords, y_coords, t_coords])

def get_batches(key: jax.random.PRNGKey, data: jnp.ndarray, batch_size: int) -> list:
    """Shuffle and split data into batches."""
    data = jax.random.permutation(key, data, axis=0)
    return [data[i:i + batch_size].astype(DTYPE) for i in range(0, data.shape[0], batch_size)]

def sample_parameters(key: jax.random.PRNGKey, param_bounds: dict[str, tuple[float, float]], n_samples: int) -> tuple[jnp.ndarray, tuple[str, ...]]:
    """
    Samples parameters from given bounds using a dictionary.

    Args:
        key: JAX PRNGKey.
        param_bounds: Dictionary mapping parameter names to (min, max) bounds.
        n_samples: The number of parameter sets to sample.

    Returns:
        A tuple of (samples_array, param_names):
        - samples_array: A (n_samples, n_params) array of sampled values.
        - param_names: A tuple of parameter names in the order of the array columns.
    """
    if not param_bounds:
        raise ValueError("param_bounds must contain at least one parameter.")
    
    names = tuple(param_bounds.keys())
    subkeys = random.split(key, len(names))
    columns = []
    
    print(f"[Data] Sampling {n_samples} sets for parameters: {names}")
    
    for (lower, upper), subkey in zip(param_bounds.values(), subkeys):
        if lower == upper:
            # Handle fixed parameter
            columns.append(jnp.full((n_samples, 1), lower, dtype=DTYPE))
        else:
            # Sample from uniform distribution
            columns.append(random.uniform(subkey, (n_samples, 1), minval=lower, maxval=upper, dtype=DTYPE))
            
    return jnp.hstack(columns), names
