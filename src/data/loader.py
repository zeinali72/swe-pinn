"""Config-driven data loading with LHS subsampling for training.

Replaces the pre-baked ``generate_training_data.py`` workflow.  Instead of
creating a fixed-size ``train_lhs_points.npy`` offline, this module loads the
full domain file at startup, selects ``n_train_samples`` points via Latin
Hypercube Sampling, and immediately frees the large source array.
"""
import gc
import os
from typing import Optional

import numpy as np
import jax.numpy as jnp

from src.config import get_dtype


def sample_training_data(
    source_path: str,
    n_samples: int,
    seed: int = 42,
    max_time: Optional[float] = None,
    verbose: bool = True,
) -> jnp.ndarray:
    """Load a full-domain data file and return an LHS subsample.

    Parameters
    ----------
    source_path : str
        Path to the full spatiotemporal field (``.npy``, columns
        ``[t, x, y, h, u, v]``).
    n_samples : int
        Number of points to select via Latin Hypercube Sampling.
    seed : int
        Random seed for reproducibility.
    max_time : float, optional
        If given, only rows with ``t <= max_time`` are eligible for sampling.
    verbose : bool
        Print progress messages.

    Returns
    -------
    jnp.ndarray, shape (n_samples, 6)
        Subsampled data in ``[t, x, y, h, u, v]`` format.
    """
    if verbose:
        print(f"Loading source data from: {source_path}")
    raw = np.load(source_path)
    if verbose:
        print(f"  Source shape: {raw.shape} ({raw.nbytes / 1e6:.1f} MB)")

    # Optional time filter
    if max_time is not None:
        mask = raw[:, 0] <= max_time
        raw = raw[mask]
        if verbose:
            print(f"  After time filter (t <= {max_time}): {raw.shape[0]} rows")

    n_available = raw.shape[0]
    if n_samples >= n_available:
        if verbose:
            print(f"  Requested {n_samples} samples but only {n_available} available. Using all.")
        result = jnp.array(raw, dtype=get_dtype())
        del raw
        gc.collect()
        return result

    # Latin Hypercube Sampling: stratified index selection
    indices = _lhs_indices(n_available, n_samples, seed)
    sample = raw[indices]

    # Free the large source array
    del raw
    gc.collect()

    result = jnp.array(sample, dtype=get_dtype())
    if verbose:
        print(f"  LHS sampled {n_samples} / {n_available} points (seed={seed})")
    return result


def _lhs_indices(n_total: int, n_samples: int, seed: int) -> np.ndarray:
    """Select indices via 1-D Latin Hypercube Sampling.

    Divides the index range ``[0, n_total)`` into ``n_samples`` equal strata
    and picks one random index per stratum.  This ensures even coverage across
    the dataset regardless of its internal ordering.
    """
    rng = np.random.default_rng(seed)
    strata = np.linspace(0, n_total, n_samples + 1, dtype=np.float64)
    lo = strata[:-1].astype(np.intp)
    hi = np.clip(strata[1:].astype(np.intp), lo + 1, n_total)
    return rng.integers(lo, hi)


def resolve_training_data(
    cfg,
    base_data_path: str,
    has_data_loss: bool,
    static_weights_dict: dict,
    *,
    verbose: bool = True,
):
    """Config-driven training data resolution.

    Tries the LHS-from-source path first (``data.source_file`` +
    ``data.n_train_samples``).  Falls back to loading a pre-sampled file
    (``data.training_file`` or default ``train_lhs_points.npy``).

    Parameters
    ----------
    cfg : dict or FrozenDict
        Experiment configuration.
    base_data_path : str
        Root path to the experiment's data directory.
    has_data_loss : bool
        Whether data-driven loss is active.
    static_weights_dict : dict
        Loss weight dict (used for informational logging only).
    verbose : bool
        Print progress messages.

    Returns
    -------
    tuple of (data_points_full, has_data_loss, data_free)
    """
    data_points_full = None

    if not has_data_loss:
        return None, False, True

    data_cfg = cfg.get("data", {})
    source_file = data_cfg.get("source_file")
    n_train_samples = data_cfg.get("n_train_samples")
    training_seed = cfg.get("training", {}).get("seed", 42)
    max_time = data_cfg.get("train_max_time")

    # --- Path 1: LHS from full-domain source file ---
    if source_file and n_train_samples:
        source_path = os.path.join(base_data_path, source_file)
        if os.path.exists(source_path):
            try:
                data_points_full = sample_training_data(
                    source_path,
                    n_train_samples,
                    seed=training_seed,
                    max_time=max_time,
                    verbose=verbose,
                )
                if verbose:
                    data_weight = static_weights_dict.get("data", 0.0)
                    print(
                        f"Using {data_points_full.shape[0]} LHS-sampled points "
                        f"for data loss term (weight={data_weight:.2e})."
                    )
                    if data_weight == 0.0:
                        print(
                            "Warning: 'data_free: false' but 'data_weight' is 0. "
                            "Data will be loaded but loss term will be 0."
                        )
                return data_points_full, True, False
            except Exception as e:
                if verbose:
                    print(f"Error sampling from source file {source_path}: {e}")
                    print("Falling back to pre-sampled training file.")
        elif verbose:
            print(
                f"Source file not found at {source_path}. "
                "Falling back to pre-sampled training file."
            )

    # --- Path 2: Load pre-sampled file (backward compatible) ---
    fallback_filename = data_cfg.get("training_file", "train_lhs_points.npy")
    training_data_file = os.path.join(base_data_path, fallback_filename)

    if os.path.exists(training_data_file):
        try:
            if verbose:
                print(f"Loading TRAINING data from: {training_data_file}")
            data_points_full = jnp.load(training_data_file).astype(get_dtype())
            if data_points_full.shape[0] == 0:
                if verbose:
                    print("Warning: Training data file is empty. Disabling data loss.")
                data_points_full = None
                has_data_loss = False
            else:
                if verbose:
                    data_weight = static_weights_dict.get("data", 0.0)
                    print(
                        f"Using {data_points_full.shape[0]} points for data loss "
                        f"term (weight={data_weight:.2e})."
                    )
                    if data_weight == 0.0:
                        print(
                            "Warning: 'data_free: false' but 'data_weight' is 0. "
                            "Data will be loaded but loss term will be 0."
                        )
        except Exception as e:
            if verbose:
                print(f"Error loading training data file {training_data_file}: {e}")
                print("Disabling data loss term due to loading error.")
            data_points_full = None
            has_data_loss = False
    else:
        if verbose:
            print(f"Warning: Training data file not found at {training_data_file}.")
            print("Data loss term cannot be computed and will be disabled.")
        has_data_loss = False

    data_free = not has_data_loss
    return data_points_full, has_data_loss, data_free
