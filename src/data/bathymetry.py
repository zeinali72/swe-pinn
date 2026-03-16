"""Differentiable bathymetry interpolation from DEM files."""
import warnings

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.ndimage
from jax import grad, jit, vmap

from src.config import get_dtype

# Module-level state for the JIT-compiled interpolator.
# Kept for backward compatibility; prefer the return value of load_bathymetry().
_BATHYMETRY_FN = None
_BATHYMETRY_WARNING_EMITTED = False


def load_bathymetry(dem_path: str):
    """Load a DEM file and create a differentiable bathymetry interpolator.

    The compiled interpolator is both stored in the module-level
    ``_BATHYMETRY_FN`` (for backward compatibility) **and** returned so that
    callers can hold an explicit reference without relying on mutable global
    state.

    Returns
    -------
    callable or None
        A vmapped function ``(x, y) -> (z, dz_dx, dz_dy)`` or ``None`` if
        the DEM file was not found.
    """
    global _BATHYMETRY_FN

    header = {}
    try:
        with open(dem_path, 'r') as f:
            for _ in range(6):
                line = f.readline().split()
                header[line[0].lower()] = float(line[1])
        dem_numpy = np.loadtxt(dem_path, skiprows=6)
    except FileNotFoundError:
        print(f"Warning: DEM file not found at {dem_path}. Bathymetry will be flat.")
        return None

    xll = header['xllcorner']
    yll = header['yllcorner']
    cellsize = header['cellsize']
    nrows = int(header['nrows'])
    dem_jax = jnp.array(dem_numpy, dtype=get_dtype())

    def get_elevation_scalar(x, y):
        col_idx = (x - xll) / cellsize
        y_max = yll + nrows * cellsize
        row_idx = (y_max - y) / cellsize
        coords = jnp.array([row_idx, col_idx])
        return jax.scipy.ndimage.map_coordinates(dem_jax, coords, order=1, mode='nearest')

    grad_z = grad(get_elevation_scalar, argnums=(0, 1))

    @jit
    def bathymetry_fn_point(x, y):
        z = get_elevation_scalar(x, y)
        dz_dx, dz_dy = grad_z(x, y)
        return z, dz_dx, dz_dy

    fn = vmap(bathymetry_fn_point)
    _BATHYMETRY_FN = fn
    print(f"Bathymetry loaded from {dem_path}")
    return fn


def get_bathymetry_fn():
    """Return the currently registered global bathymetry function.

    Unlike importing ``bathymetry_fn`` at the module level, this always
    reflects the latest ``load_bathymetry`` call.
    """
    return _BATHYMETRY_FN


def bathymetry_fn(x, y):
    """Public accessor for the bathymetry function.

    Returns (z, dz_dx, dz_dy) for each point.
    Falls back to flat domain (z=0) if no DEM is loaded.
    """
    global _BATHYMETRY_WARNING_EMITTED
    if _BATHYMETRY_FN is None:
        if not _BATHYMETRY_WARNING_EMITTED:
            warnings.warn(
                "Bathymetry not loaded — using flat domain (z=0). "
                "Call load_bathymetry() first if terrain is expected.",
                stacklevel=2
            )
            _BATHYMETRY_WARNING_EMITTED = True
        return jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)
    return _BATHYMETRY_FN(x, y)


def reset_bathymetry():
    """Reset module-level bathymetry state.

    Useful in tests or HPO loops where multiple configs are loaded in the
    same process and stale state from a previous run must be cleared.
    """
    global _BATHYMETRY_FN, _BATHYMETRY_WARNING_EMITTED
    _BATHYMETRY_FN = None
    _BATHYMETRY_WARNING_EMITTED = False
