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

    When total_batches > n_batches_avail, the data is re-shuffled for
    each cycle so that later batches are fresh permutations rather than
    exact duplicates of earlier ones.
    """
    n_samples = data.shape[0]
    n_batches_avail = n_samples // batch_size
    # Number of full cycles needed to cover total_batches.
    n_cycles = -((-total_batches) // n_batches_avail)  # ceil division

    # Generate a different PRNG key per cycle.
    keys = random.split(key, n_cycles)

    def _shuffle_one(k):
        shuffled = random.permutation(k, data, axis=0)
        shuffled = shuffled[:n_batches_avail * batch_size]
        return shuffled.reshape((n_batches_avail, batch_size, -1))

    # Stack independently-shuffled cycles: (n_cycles, n_batches_avail, batch_size, features)
    all_cycles = jax.vmap(_shuffle_one)(keys)
    # Flatten cycles: (n_cycles * n_batches_avail, batch_size, features)
    all_batches = all_cycles.reshape((-1, batch_size, data.shape[-1]))
    return all_batches[:total_batches]


def get_sample_count(sampling_cfg, name, default):
    """Helper to safely get sample counts from config."""
    return sampling_cfg.get(name, default)
