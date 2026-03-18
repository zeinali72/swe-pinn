"""Computational cost metrics: D2-D4."""
import time
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np


@contextmanager
def timed(label: str = ""):
    """Context manager that records elapsed wall-clock time.

    Yields a dict populated with 'elapsed_s' (and 'label') on exit.
    """
    info = {"label": label, "elapsed_s": None}
    t0 = time.perf_counter()
    try:
        yield info
    finally:
        info["elapsed_s"] = time.perf_counter() - t0


def training_cost(start_time: float, end_time: float) -> dict:
    """D2: Wall-clock training cost.

    Args:
        start_time: time.perf_counter() value at training start.
        end_time: time.perf_counter() value at training end.

    Returns:
        Dict with 'elapsed_s' and 'label'.
    """
    return {"elapsed_s": end_time - start_time, "label": "training"}


def inference_cost(model, params: dict, coords: np.ndarray, batch_size: int = 50_000) -> dict:
    """D3: Inference cost for a single forward pass over the evaluation grid.

    Runs prediction in batches, times total wall-clock duration (including JAX block_until_ready).

    Args:
        model: Flax model.
        params: Parameter dict with 'params' key.
        coords: (N, 3) evaluation coordinates.
        batch_size: Batch size for chunked inference.

    Returns:
        Dict with 'elapsed_s', 'n_points', 'throughput_pts_per_s'.
    """
    coords = jnp.asarray(coords)
    n_points = coords.shape[0]
    flax_params = {"params": params["params"]}

    # Warm-up pass to avoid measuring JIT compile time
    _ = model.apply(flax_params, coords[:min(8, n_points)], train=False)
    jax.block_until_ready(_)

    t0 = time.perf_counter()
    outputs = []
    for start in range(0, n_points, batch_size):
        batch = coords[start: start + batch_size]
        out = model.apply(flax_params, batch, train=False)
        outputs.append(out)
    jax.block_until_ready(outputs[-1])
    elapsed = time.perf_counter() - t0

    throughput = n_points / elapsed if elapsed > 0 else float("inf")
    return {
        "elapsed_s": elapsed,
        "n_points": n_points,
        "throughput_pts_per_s": throughput,
    }


def break_even_query_count(
    t_data_prep: float,
    t_training: float,
    t_inference: float,
    t_icm: float,
) -> dict:
    """D4: Break-even number of queries N_break.

    N_break = (T_data_prep + T_training) / (T_ICM - T_inference)

    If T_ICM <= T_inference, the PINN is never faster; returns inf.

    Returns:
        Dict with 'n_break', 't_data_prep', 't_training', 't_inference', 't_icm'.
    """
    upfront = t_data_prep + t_training
    saving_per_query = t_icm - t_inference
    if saving_per_query <= 0:
        n_break = float("inf")
    else:
        n_break = int(np.ceil(upfront / saving_per_query))

    return {
        "n_break": n_break,
        "t_data_prep": t_data_prep,
        "t_training": t_training,
        "t_inference": t_inference,
        "t_icm": t_icm,
    }
