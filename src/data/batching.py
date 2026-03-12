"""Batching utilities for training data."""
import jax
import jax.numpy as jnp
from jax import random

from src.config import DTYPE


def get_batches(key: jax.random.PRNGKey, data: jnp.ndarray, batch_size: int) -> list:
    """Shuffle and split data into batches, dropping the remainder."""
    data = jax.random.permutation(key, data, axis=0)
    num_batches = data.shape[0] // batch_size
    if num_batches == 0:
        return []
    return [data[i * batch_size : (i + 1) * batch_size].astype(DTYPE)
            for i in range(num_batches)]


def get_batches_tensor(key, data, batch_size, total_batches):
    """JIT-compatible batching helper.

    Returns (total_batches, batch_size, features).
    Assumes data.shape[0] >= batch_size (checked by caller).
    """
    n_samples = data.shape[0]
    n_batches_avail = n_samples // batch_size
    data = random.permutation(key, data, axis=0)
    data = data[:n_batches_avail * batch_size]
    data = data.reshape((n_batches_avail, batch_size, -1))
    indices = jnp.arange(total_batches) % n_batches_avail
    return data[indices]


def get_sample_count(sampling_cfg, name, default):
    """Helper to safely get sample counts from config."""
    return sampling_cfg.get(name, default)
