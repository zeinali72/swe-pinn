"""Data-mode resolution and training/validation data loading — shared across experiments."""
import os
import sys

import jax.numpy as jnp

from src.config import get_dtype
from src.data import load_validation_data, load_bathymetry, load_boundary_condition
from src.data.loader import resolve_training_data  # noqa: F401 — re-exported
from src.training.setup import resolve_configured_asset_path


def resolve_data_mode(cfg, verbose=True):
    """Determine whether training is data-free or data-driven.

    Returns
    -------
    (data_free, has_data_loss)
        data_free : bool — True means physics-only
        has_data_loss : bool — True means data loss will be computed
    """
    data_free_flag = cfg.get("data_free")

    if data_free_flag is False:
        if verbose:
            print("Info: 'data_free: false' found in config. Activating data-driven mode.")
        return False, True
    else:
        if verbose:
            if data_free_flag is None:
                print("Warning: 'data_free' flag not specified in config. Defaulting to 'data_free: true'.")
            else:
                print("Info: 'data_free: true' found in config. Data loss term will be disabled.")
        return True, False


def load_training_data(base_data_path, has_data_loss, static_weights_dict,
                       filename="train_lhs_points.npy", verbose=True):
    """Load the training data .npy file if data-driven mode is active.

    Returns
    -------
    (data_points_full, has_data_loss, data_free)
        data_points_full may be None even when initially has_data_loss=True
        if the file is missing or empty.
    """
    data_points_full = None
    training_data_file = os.path.join(base_data_path, filename)

    if has_data_loss:
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
                        data_weight = static_weights_dict.get('data', 0.0)
                        print(f"Using {data_points_full.shape[0]} points for data loss term (weight={data_weight:.2e}).")
                        if data_weight == 0.0:
                            print("Warning: 'data_free: false' but 'data_weight' is 0. Data will be loaded but loss term will be 0.")
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


def load_validation_from_file(base_data_path, filename="val_gauges_gt.npy", verbose=True):
    """Load validation gauge data if it exists.

    Returns
    -------
    dict with keys:
        loaded : bool
        full_val_data : ndarray | None
        val_points : ndarray | None
        h_true_val : ndarray | None
    """
    result = {"loaded": False, "full_val_data": None, "val_points": None, "h_true_val": None, "val_targets": None}
    validation_data_file = os.path.join(base_data_path, filename)

    if not os.path.exists(validation_data_file):
        if verbose:
            print(f"Warning: Validation data not found at {validation_data_file}. Skipping dense validation.")
        return result

    try:
        if verbose:
            print(f"Loading VALIDATION data from: {validation_data_file}")
        full_val_data, val_points, val_targets = load_validation_data(validation_data_file, dtype=get_dtype())
        h_true_val = val_targets[:, 0]
        if val_points.shape[0] > 0:
            result.update({
                "loaded": True,
                "full_val_data": full_val_data,
                "val_points": val_points,
                "h_true_val": h_true_val,
                "val_targets": val_targets,
            })
        else:
            if verbose:
                print("Warning: No validation points remaining after masking. NSE/RMSE calculation will be skipped.")
    except Exception as e:
        if verbose:
            print(f"Error loading or processing validation data file {validation_data_file}: {e}")
            print("NSE/RMSE calculation using loaded data will be skipped.")

    return result


def load_terrain_assets(cfg, base_data_path, scenario_name, *, load_dem=True, verbose=True):
    """Load DEM (bathymetry) and boundary-condition CSV assets for terrain experiments.

    This consolidates the repeated asset-loading pattern used across experiments 3-7.
    Each experiment resolves the configured asset path, loads the DEM into the global
    bathymetry interpolator, and creates a JIT-compatible boundary-condition function
    from the CSV time-series.

    Parameters
    ----------
    cfg : dict or FrozenDict
        Experiment configuration (must contain ``data.assets`` section).
    base_data_path : str
        Root path to the experiment's data directory.
    scenario_name : str
        Name of the scenario sub-directory inside *base_data_path*.
    load_dem : bool, optional
        If True (default), load bathymetry from DEM file.  Set to False for
        experiments that do not use terrain data (e.g. Experiment 6).
    verbose : bool, optional
        If False, suppress informational print statements (useful for HPO runs).

    Returns
    -------
    dict
        ``"bc_fn"`` — JIT-compatible boundary-condition interpolation function.
        ``"dem_path"`` — resolved DEM file path (None when *load_dem* is False).
        ``"bc_csv_path"`` — resolved boundary-condition CSV path.
    """
    dem_path = None
    if load_dem:
        try:
            dem_path = resolve_configured_asset_path(cfg, base_data_path, scenario_name, "dem")
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        if verbose:
            print(f"Loading Bathymetry from {dem_path}...")
        load_bathymetry(dem_path)

    try:
        bc_csv_path = resolve_configured_asset_path(cfg, base_data_path, scenario_name, "boundary_condition")
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    bc_fn = load_boundary_condition(bc_csv_path)

    return {
        "bc_fn": bc_fn,
        "dem_path": dem_path,
        "bc_csv_path": bc_csv_path,
    }
