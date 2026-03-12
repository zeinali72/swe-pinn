"""Shared epoch-data generation helpers and scan-body factory.

Provides ``make_scan_body`` to eliminate repeated scan-body closures, and
``sample_and_batch`` / ``empty_batch`` helpers to reduce boilerplate in
per-experiment ``generate_epoch_data`` functions.
"""
from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
from jax import random

from src.config import DTYPE
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
    extra_static_args: tuple = (),
) -> Callable:
    """Return a ``scan_body(carry, batch_data)`` closure ready for ``lax.scan``.

    The returned function passes *batch_data* directly to *train_step_fn*,
    so ``generate_epoch_data`` must produce a **flat** dict whose keys match
    what *train_step_fn* expects (e.g. ``'bc_left'`` not ``{'bc': {'left': ...}}``).

    Parameters
    ----------
    train_step_fn : callable
        The **JIT-compiled** training step.  Its positional signature must be
        one of the two patterns used in the codebase:

        *   ``(model, params, opt_state, all_batches, weights_dict, optimiser, config, data_free)``
            — used by experiments 1 & 2.
        *   ``(model, optimiser, params, opt_state, batch, config, data_free, *extra_static_args, weights_dict)``
            — used by experiments 3–8 (includes ``bc_fn_static`` etc.).

        Because different experiments have different positional signatures,
        pass the correct *extra_static_args* tuple and the factory will
        build the right lambda.
    extra_static_args : tuple
        Additional static arguments inserted between ``data_free`` and
        ``weights_dict`` (e.g. ``(bc_fn_static,)`` for experiments 3–8).
    """

    if extra_static_args:
        # Experiments 3-8 signature:
        # train_step(model, optimiser, params, opt_state, batch, config, data_free, *extra, weights_dict)
        def scan_body(carry, batch_data):
            curr_params, curr_opt_state = carry
            new_params, new_opt_state, terms, total = train_step_fn(
                model, optimiser, curr_params, curr_opt_state,
                batch_data, config, data_free, *extra_static_args, weights_dict,
            )
            return (new_params, new_opt_state), (terms, total)
    else:
        # Experiments 1-2 signature:
        # train_step(model, params, opt_state, all_batches, weights_dict, optimiser, config, data_free)
        def scan_body(carry, batch_data):
            curr_params, curr_opt_state = carry
            new_params, new_opt_state, terms, total = train_step_fn(
                model, curr_params, curr_opt_state,
                batch_data, weights_dict, optimiser, config, data_free,
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
    return jnp.zeros((num_batches, 0, feature_dim), dtype=DTYPE)


def empty_batch(num_batches: int, feature_dim: int = 3) -> jnp.ndarray:
    """Return a zero-size placeholder batch ``(num_batches, 0, feature_dim)``."""
    return jnp.zeros((num_batches, 0, feature_dim), dtype=DTYPE)


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
    return jnp.zeros((num_batches, 0, 6), dtype=DTYPE)
