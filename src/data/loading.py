"""Data loading utilities for validation and boundary conditions."""
import numpy as np
import jax.numpy as jnp


def load_validation_data(path: str, dtype=None):
    """Load validation data, reordering columns from [t,x,y,...] to [x,y,t].

    Returns:
        raw_data: Full loaded array (for plotting)
        inputs: Reordered coordinates [x, y, t]
        targets: Target values [h] or [h, hu, hv], depending on file format
    """
    data = np.load(path)
    assert data.ndim == 2, f"{path}: Expected 2D array, got {data.ndim}D"
    assert data.shape[1] >= 4, (
        f"{path}: Expected >=4 columns [t,x,y,h] or [t,x,y,h,hu,hv], got {data.shape[1]}"
    )
    if dtype is not None:
        data = data.astype(dtype)
    inputs = data[:, [1, 2, 0]]   # [x, y, t]
    targets = data[:, 3:6] if data.shape[1] >= 6 else data[:, 3:4]
    return data, inputs, targets


def load_boundary_condition(csv_path):
    """Read a boundary condition CSV and return a JIT-compatible interpolation function.

    Expected format: Time (mins), Water level (m)
    """
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    t_data_seconds = data[:, 0] * 60.0
    h_data_meters = data[:, 1]

    t_ref = jnp.array(t_data_seconds)
    h_ref = jnp.array(h_data_meters)

    def get_bc_h(t):
        """Input: t (scalar or vector in seconds). Output: Water level h (meters)."""
        return jnp.interp(t, t_ref, h_ref)

    return get_bc_h
