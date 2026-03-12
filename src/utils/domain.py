"""Domain geometry utilities."""
from collections.abc import Mapping

import jax.numpy as jnp


def mask_points_inside_building(points: jnp.ndarray, building_config) -> jnp.ndarray:
    """Create a boolean mask to exclude points inside a building's footprint.

    Points shape: (N, 3) where columns are (x, y, t).
    Returns: Boolean array of shape (N,), True for points OUTSIDE.
    """
    if not isinstance(building_config, Mapping) or not building_config:
        return jnp.ones((points.shape[0],), dtype=bool)

    x_coords = points[:, 0]
    y_coords = points[:, 1]

    x_min = building_config["x_min"]
    x_max = building_config["x_max"]
    y_min = building_config["y_min"]
    y_max = building_config["y_max"]

    mask = ~((x_coords >= x_min) & (x_coords <= x_max) &
             (y_coords >= y_min) & (y_coords <= y_max))
    return mask
