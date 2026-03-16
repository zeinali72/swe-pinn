"""Differentiable bathymetry interpolation from DEM files."""
import warnings

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.ndimage
from jax import grad, jit, vmap

from src.config import DTYPE

# Module-level state for the JIT-compiled interpolator.
_BATHYMETRY_FN = None
_BATHYMETRY_WARNING_EMITTED = False


def load_bathymetry(dem_path: str):
    """Load a DEM file and create a differentiable bathymetry interpolator.

    Call this ONCE at the start of your training script.
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
        return

    xll = header['xllcorner']
    yll = header['yllcorner']
    cellsize = header['cellsize']
    nrows = int(header['nrows'])
    dem_jax = jnp.array(dem_numpy, dtype=DTYPE)

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

    _BATHYMETRY_FN = vmap(bathymetry_fn_point)
    print(f"Bathymetry loaded from {dem_path}")


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
