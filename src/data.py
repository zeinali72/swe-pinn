# src/data.py
import jax
import jax.numpy as jnp
from jax import random
from flax.core import FrozenDict
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
    """Shuffle and split data into batches, dropping the remainder."""
    data = jax.random.permutation(key, data, axis=0)
    
    # Calculate the number of full batches using integer division
    num_batches = data.shape[0] // batch_size
    
    # If no full batches can be made, return an empty list
    if num_batches == 0:
        return []

    # Iterate only up to the last full batch
    # This guarantees every batch is of size `batch_size`
    return [data[i * batch_size : (i + 1) * batch_size].astype(DTYPE) 
            for i in range(num_batches)]

def sample_domain(key: jax.random.PRNGKey, n_points: int,
                  x_bounds: tuple, y_bounds: tuple, t_bounds: tuple) -> jnp.ndarray:
    """Sample points uniformly in a domain defined by (x, y, t) bounds."""
    if n_points <= 0:
        return jnp.empty((0, 3), dtype=DTYPE)

    key_x, key_y, key_t = random.split(key, 3)
    x_start, x_end = x_bounds
    y_start, y_end = y_bounds
    t_start, t_end = t_bounds

    x_coords = jnp.full((n_points, 1), x_start, dtype=DTYPE) if x_start == x_end else \
        random.uniform(key_x, (n_points, 1), minval=x_start, maxval=x_end, dtype=DTYPE)
    y_coords = jnp.full((n_points, 1), y_start, dtype=DTYPE) if y_start == y_end else \
        random.uniform(key_y, (n_points, 1), minval=y_start, maxval=y_end, dtype=DTYPE)
    t_coords = jnp.full((n_points, 1), t_start, dtype=DTYPE) if t_start == t_end else \
        random.uniform(key_t, (n_points, 1), minval=t_start, maxval=t_end, dtype=DTYPE)

    return jnp.hstack([x_coords, y_coords, t_coords])
