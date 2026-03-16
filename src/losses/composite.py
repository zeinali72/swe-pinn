"""Composite loss wrappers for standard rectangular domains."""
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict
from typing import Dict, Any, Optional, Callable

from src.physics import h_exact
from src.losses.boundary import (
    loss_boundary_dirichlet_h,
    loss_boundary_dirichlet_hu,
    loss_boundary_neumann_outflow_x,
    loss_boundary_wall_horizontal,
    loss_boundary_wall_vertical,
)


def compute_bc_loss(model: nn.Module, params: Dict[str, Any],
                    left_batch: jnp.ndarray, right_batch: jnp.ndarray,
                    bottom_batch: jnp.ndarray, top_batch: jnp.ndarray,
                    config: FrozenDict,
                    bc_fn: Optional[Callable] = None) -> jnp.ndarray:
    """Compose atomic losses for standard rectangular domains."""
    # Left Boundary
    if bc_fn is not None:
        t_left = left_batch[..., 2]
        h_target = bc_fn(t_left)
        loss_left = loss_boundary_dirichlet_h(model, params, left_batch, h_target)
    else:
        u_const = config["physics"]["u_const"]
        n_manning = config["physics"]["n_manning"]
        t_left = left_batch[..., 2]
        h_true = h_exact(0.0, t_left, n_manning, u_const)
        hu_true = h_true * u_const
        loss_left = (loss_boundary_dirichlet_h(model, params, left_batch, h_true) +
                     loss_boundary_dirichlet_hu(model, params, left_batch, hu_true))

    # Right Boundary (Neumann Outflow)
    loss_right = loss_boundary_neumann_outflow_x(model, params, right_batch)

    # Top/Bottom (Horizontal Walls)
    loss_bottom = loss_boundary_wall_horizontal(model, params, bottom_batch)
    loss_top = loss_boundary_wall_horizontal(model, params, top_batch)

    return loss_left + loss_right + loss_bottom + loss_top


def compute_building_bc_loss(model: nn.Module, params: Dict[str, Any],
                             building_left_batch: jnp.ndarray,
                             building_right_batch: jnp.ndarray,
                             building_bottom_batch: jnp.ndarray,
                             building_top_batch: jnp.ndarray) -> jnp.ndarray:
    """Compute slip loss for a rectangular building obstacle."""
    loss_left = loss_boundary_wall_vertical(model, params, building_left_batch)
    loss_right = loss_boundary_wall_vertical(model, params, building_right_batch)
    loss_bottom = loss_boundary_wall_horizontal(model, params, building_bottom_batch)
    loss_top = loss_boundary_wall_horizontal(model, params, building_top_batch)
    return loss_left + loss_right + loss_bottom + loss_top


def total_loss(terms: Dict[str, jnp.ndarray], weights: Dict[str, float]) -> jnp.ndarray:
    """Combine weighted loss terms into a single scalar loss."""
    loss = 0.0
    for key in terms.keys():
        if key in weights:
            loss += weights[key] * terms[key]
    return loss
