"""Conservation metrics: volume balance (B1) and continuity residual (B3)."""
import numpy as np
import jax.numpy as jnp

from src.metrics.conservation import volume_balance as _volume_balance
from src.metrics.conservation import continuity_residual as _continuity_residual


def compute_volume_balance(
    h_pred: np.ndarray,
    cell_areas: np.ndarray,
    inflow_volume: np.ndarray,
    outflow_volume: np.ndarray,
    domain_bounds: dict = None,
    coords: np.ndarray = None,
) -> dict:
    """B1: Volume balance error over time.

    Args:
        h_pred: (n_times, n_spatial) or (N,) flat array of predicted water depths.
        cell_areas: (n_spatial,) cell areas, or None for uniform rectangular cells.
        inflow_volume: (n_times,) cumulative inflow volume.
        outflow_volume: (n_times,) cumulative outflow volume.
        domain_bounds: Dict with 'lx', 'ly', 't_final' (used if cell_areas is None).

    Returns:
        Dict with 'e_mass' (array of errors per timestep), 'max_error' (float),
        'final_error' (float).
    """
    h_pred = jnp.asarray(h_pred)

    if h_pred.ndim == 2 and inflow_volume is None:
        inflow_volume = jnp.zeros(h_pred.shape[0])
    if h_pred.ndim == 2 and outflow_volume is None:
        outflow_volume = jnp.zeros(h_pred.shape[0])

    if h_pred.ndim == 2:
        n_times, n_spatial = h_pred.shape
        inflow_volume = jnp.asarray(inflow_volume)
        outflow_volume = jnp.asarray(outflow_volume)
        if cell_areas is not None:
            areas = jnp.asarray(cell_areas)
            domain_volume = jnp.sum(h_pred * areas[None, :], axis=1)
        else:
            lx = domain_bounds["lx"]
            ly = domain_bounds["ly"]
            cell_area = lx * ly / n_spatial
            domain_volume = jnp.sum(h_pred, axis=1) * cell_area
        v0 = domain_volume[0]
        expected_volume = v0 + inflow_volume - outflow_volume
        e_mass = domain_volume - expected_volume
    else:
        # Flat array — delegate to src implementation (needs coords for time-binning)
        if domain_bounds is None:
            raise ValueError("domain_bounds required when h_pred is 1D")
        if coords is None:
            raise ValueError("coords required when h_pred is 1D (needed for time-binning)")
        result = _volume_balance(jnp.asarray(h_pred), jnp.asarray(coords), domain_bounds)
        vol_series = jnp.array([v for _, v in result["volume_time_series"]])
        v0 = vol_series[0]
        e_mass = vol_series - v0
        return {
            "e_mass": np.array(e_mass),
            "max_error": float(jnp.max(jnp.abs(e_mass))),
            "final_error": float(e_mass[-1]),
        }

    return {
        "e_mass": np.array(e_mass),
        "max_error": float(jnp.max(jnp.abs(e_mass))),
        "final_error": float(e_mass[-1]),
    }


def compute_continuity_residual(model, params: dict, coords: np.ndarray, config: dict) -> dict:
    """B3: Continuity residual dh/dt + d(hu)/dx + d(hv)/dy via autodiff."""
    return _continuity_residual(model, params, jnp.asarray(coords), config)
