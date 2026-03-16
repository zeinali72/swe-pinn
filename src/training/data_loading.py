"""Data-mode resolution and training/validation data loading — shared across experiments."""
import os
import sys

import jax.numpy as jnp

from src.config import get_dtype
from src.data import load_validation_data


def resolve_data_mode(cfg):
    """Determine whether training is data-free or data-driven.

    Returns
    -------
    (data_free, has_data_loss)
        data_free : bool — True means physics-only
        has_data_loss : bool — True means data loss will be computed
    """
    data_free_flag = cfg.get("data_free")

    if data_free_flag is False:
        print("Info: 'data_free: false' found in config. Activating data-driven mode.")
        return False, True
    else:
        if data_free_flag is None:
            print("Warning: 'data_free' flag not specified in config. Defaulting to 'data_free: true'.")
        else:
            print("Info: 'data_free: true' found in config. Data loss term will be disabled.")
        return True, False


def load_training_data(base_data_path, has_data_loss, static_weights_dict,
                       filename="training_dataset_sample.npy"):
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
                print(f"Loading TRAINING data from: {training_data_file}")
                data_points_full = jnp.load(training_data_file).astype(get_dtype())
                if data_points_full.shape[0] == 0:
                    print("Warning: Training data file is empty. Disabling data loss.")
                    data_points_full = None
                    has_data_loss = False
                else:
                    data_weight = static_weights_dict.get('data', 0.0)
                    print(f"Using {data_points_full.shape[0]} points for data loss term (weight={data_weight:.2e}).")
                    if data_weight == 0.0:
                        print("Warning: 'data_free: false' but 'data_weight' is 0. Data will be loaded but loss term will be 0.")
            except Exception as e:
                print(f"Error loading training data file {training_data_file}: {e}")
                print("Disabling data loss term due to loading error.")
                data_points_full = None
                has_data_loss = False
        else:
            print(f"Warning: Training data file not found at {training_data_file}.")
            print("Data loss term cannot be computed and will be disabled.")
            has_data_loss = False

    data_free = not has_data_loss
    return data_points_full, has_data_loss, data_free


def load_validation_from_file(base_data_path, filename="validation_gauges.npy"):
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
        print(f"Warning: Validation data not found at {validation_data_file}. Skipping dense validation.")
        return result

    try:
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
            print("Warning: No validation points remaining after masking. NSE/RMSE calculation will be skipped.")
    except Exception as e:
        print(f"Error loading or processing validation data file {validation_data_file}: {e}")
        print("NSE/RMSE calculation using loaded data will be skipped.")

    return result
