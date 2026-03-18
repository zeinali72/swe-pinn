"""Computational cost metrics: D1-D4.

D1: Data preparation cost (ICM wall-clock) — recorded externally, stored here.
D2: Training cost — wall-clock from epoch 1 to convergence.
D3: Inference cost — single forward pass over full evaluation grid.
D4: Break-even query count — N where PINN amortises its upfront cost vs ICM.
"""
import time
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np


@contextmanager
def timed(label: str = ""):
    """Context manager that records elapsed wall-clock time.

    Usage::

        with timed("training") as t:
            run_training_loop(...)
        print(t["elapsed_s"])

    Yields a dict that is populated with ``elapsed_s`` (and ``label``) on exit.
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
        start_time: ``time.perf_counter()`` value at training start.
        end_time:   ``time.perf_counter()`` value at training end.

    Returns:
        Dict with ``elapsed_s`` and ``label = 'training'``.
    """
    return {"elapsed_s": end_time - start_time, "label": "training"}


def inference_cost(model, params: dict, coords: np.ndarray, batch_size: int = 50_000) -> dict:
    """D3: Inference cost for a single forward pass over the evaluation grid.

    Runs a warm-up pass first to exclude JIT compilation from the measurement,
    then times the full batched forward pass including ``jax.block_until_ready``.

    Args:
        model:      Flax model.
        params:     Parameter dict (must contain ``params`` key).
        coords:     (N, 3) evaluation coordinates.
        batch_size: Chunk size for batched inference.

    Returns:
        Dict with ``elapsed_s``, ``n_points``, ``throughput_pts_per_s``.
    """
    coords = jnp.asarray(coords)
    n_points = coords.shape[0]
    flax_params = {"params": params["params"]}

    # Warm-up: avoids measuring JIT compilation
    warmup = model.apply(flax_params, coords[:min(8, n_points)], train=False)
    jax.block_until_ready(warmup)

    t0 = time.perf_counter()
    outputs = []
    for start in range(0, n_points, batch_size):
        chunk = coords[start: start + batch_size]
        outputs.append(model.apply(flax_params, chunk, train=False))
    jax.block_until_ready(outputs[-1])
    elapsed = time.perf_counter() - t0

    return {
        "elapsed_s": elapsed,
        "n_points": n_points,
        "throughput_pts_per_s": n_points / elapsed if elapsed > 0 else float("inf"),
    }


def break_even_query_count(
    t_data_prep: float,
    t_training: float,
    t_inference: float,
    t_icm: float,
) -> dict:
    """D4: Number of inference queries at which PINN breaks even against ICM.

    N_break = ceil((T_data_prep + T_training) / (T_ICM - T_inference))

    If T_ICM <= T_inference the PINN is never faster per query; returns inf.

    Returns:
        Dict with ``n_break``, ``t_data_prep``, ``t_training``,
        ``t_inference``, ``t_icm``.
    """
    upfront = t_data_prep + t_training
    saving_per_query = t_icm - t_inference
    if saving_per_query <= 0:
        n_break: float | int = float("inf")
    else:
        n_break = int(np.ceil(upfront / saving_per_query))

    return {
        "n_break": n_break,
        "t_data_prep": t_data_prep,
        "t_training": t_training,
        "t_inference": t_inference,
        "t_icm": t_icm,
    }
