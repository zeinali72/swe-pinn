"""Boundary condition losses: Dirichlet, Neumann, and slip wall."""
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Any


def loss_boundary_dirichlet_h(model: nn.Module, params: Dict[str, Any],
                              batch: jnp.ndarray,
                              h_target: jnp.ndarray) -> jnp.ndarray:
    """Enforces a prescribed water level h (constant or time-varying)."""
    U_pred = model.apply({'params': params['params']}, batch, train=False)
    h_pred = U_pred[..., 0]
    if h_target.ndim != h_pred.ndim:
        h_target = h_target.reshape(h_pred.shape)
    return jnp.mean((h_pred - h_target)**2)


def loss_boundary_dirichlet_hu(model: nn.Module, params: Dict[str, Any],
                               batch: jnp.ndarray,
                               hu_target: jnp.ndarray) -> jnp.ndarray:
    """Enforces a prescribed momentum hu (constant or time-varying)."""
    U_pred = model.apply({'params': params['params']}, batch, train=False)
    hu_pred = U_pred[..., 1]
    if hu_target.ndim != hu_pred.ndim:
        hu_target = hu_target.reshape(hu_pred.shape)
    return jnp.mean((hu_pred - hu_target)**2)


def loss_boundary_dirichlet_hv(model: nn.Module, params: Dict[str, Any],
                               batch: jnp.ndarray,
                               hv_target: jnp.ndarray) -> jnp.ndarray:
    """Enforces a prescribed momentum hv (constant or time-varying)."""
    U_pred = model.apply({'params': params['params']}, batch, train=False)
    hv_pred = U_pred[..., 2]
    if hv_target.ndim != hv_pred.ndim:
        hv_target = hv_target.reshape(hv_pred.shape)
    return jnp.mean((hv_pred - hv_target)**2)


def loss_slip_wall_generalized(model, params, batch):
    """Enforces no-flux condition: u . n = 0.

    batch: [x, y, t, nx, ny]
    """
    coords = batch[..., :3]
    normals = batch[..., 3:5]
    U = model.apply({'params': params['params']}, coords, train=False)
    hu = U[..., 1]
    hv = U[..., 2]
    flux = hu * normals[..., 0] + hv * normals[..., 1]
    return jnp.mean(flux**2)


def loss_boundary_wall_vertical(model: nn.Module, params: Dict[str, Any],
                                batch: jnp.ndarray) -> jnp.ndarray:
    """Enforces hu = 0 (no flow through X-aligned boundary)."""
    U_pred = model.apply({'params': params['params']}, batch, train=False)
    hu_pred = U_pred[..., 1]
    return jnp.mean(hu_pred**2)


def loss_boundary_wall_horizontal(model: nn.Module, params: Dict[str, Any],
                                  batch: jnp.ndarray) -> jnp.ndarray:
    """Enforces hv = 0 (no flow through Y-aligned boundary)."""
    U_pred = model.apply({'params': params['params']}, batch, train=False)
    hv_pred = U_pred[..., 2]
    return jnp.mean(hv_pred**2)


def loss_boundary_neumann_outflow_x(model: nn.Module, params: Dict[str, Any],
                                    batch: jnp.ndarray) -> jnp.ndarray:
    """Enforces zero gradient in x-direction: d(U)/dx = 0."""
    def U_fn(pts):
        return model.apply({'params': params['params']}, pts, train=False)

    jac_U = jax.vmap(jax.jacfwd(U_fn))(batch)
    dU_dx = jac_U[..., 0]
    return jnp.mean(dU_dx**2)
