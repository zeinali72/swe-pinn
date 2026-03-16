"""Shared epoch-data generation helpers and scan-body factory.

Provides ``make_scan_body`` to eliminate repeated scan-body closures, and
``sample_and_batch`` / ``empty_batch`` helpers to reduce boilerplate in
per-experiment ``generate_epoch_data`` functions.
"""
from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
from jax import random

from src.config import get_dtype
from src.data import get_batches_tensor


# ---------------------------------------------------------------------------
# scan_body factory
# ---------------------------------------------------------------------------

def make_scan_body(
    train_step_fn: Callable,
    model: Any,
    optimiser: Any,
    weights_dict: Any,
    config: Any,
    data_free: bool,
    *,
    compute_losses_fn: Callable,
) -> Callable:
    """Return a ``scan_body(carry, batch_data)`` closure ready for ``lax.scan``.

    Parameters
    ----------
    train_step_fn : callable
        The **JIT-compiled** generic training step.  Its positional signature
        must be::

            (model, optimiser, params, opt_state, batch, config,
             data_free, compute_losses_fn, weights_dict)

    compute_losses_fn : callable
        Experiment-specific loss computation function with signature::

            (model, params, batch, config, data_free) -> dict[str, scalar]
    """

    def scan_body(carry, batch_data):
        curr_params, curr_opt_state = carry
        new_params, new_opt_state, terms, total = train_step_fn(
            model, optimiser, curr_params, curr_opt_state,
            batch_data, config, data_free, compute_losses_fn, weights_dict,
        )
        return (new_params, new_opt_state), (terms, total)

    return scan_body


# ---------------------------------------------------------------------------
# Batch helpers for generate_epoch_data
# ---------------------------------------------------------------------------

def sample_and_batch(
    key: jnp.ndarray,
    sample_fn: Callable,
    n_points: int,
    batch_size: int,
    num_batches: int,
    *sample_args,
    feature_dim: int = 3,
) -> jnp.ndarray:
    """Sample *n_points* via *sample_fn* and reshape into batches.

    Returns an array of shape ``(num_batches, batch_size, feature_dim)`` or
    a zero-filled placeholder when there aren't enough points.
    """
    if n_points // batch_size > 0:
        pts = sample_fn(key, n_points, *sample_args)
        return get_batches_tensor(key, pts, batch_size, num_batches)
    return jnp.zeros((num_batches, 0, feature_dim), dtype=get_dtype())


def empty_batch(num_batches: int, feature_dim: int = 3) -> jnp.ndarray:
    """Return a zero-size placeholder batch ``(num_batches, 0, feature_dim)``."""
    return jnp.zeros((num_batches, 0, feature_dim), dtype=get_dtype())


def maybe_batch_data(
    key: jnp.ndarray,
    data_points_full,
    batch_size: int,
    num_batches: int,
    data_free: bool,
) -> jnp.ndarray:
    """Batch training data points, or return an empty placeholder if data-free."""
    if not data_free and data_points_full is not None:
        return get_batches_tensor(key, data_points_full, batch_size, num_batches)
    return jnp.zeros((num_batches, 0, 6), dtype=get_dtype())
