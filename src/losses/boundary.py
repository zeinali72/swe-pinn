"""Boundary condition losses: Dirichlet, Neumann, and slip wall."""
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Any


def loss_boundary_dirichlet(model: nn.Module, params: Dict[str, Any],
                            batch: jnp.ndarray, target: jnp.ndarray,
                            var_idx: int) -> jnp.ndarray:
    """Enforces a prescribed Dirichlet value on a single output variable.

    Args:
        model: Flax neural network module.
        params: Model parameter dictionary.
        batch: Collocation points at the boundary.
        target: Target values (constant or time-varying).
        var_idx: Output channel index (0 = h, 1 = hu, 2 = hv).
    """
    U_pred = model.apply({'params': params['params']}, batch, train=False)
    pred = U_pred[..., var_idx]
    if target.ndim != pred.ndim:
        target = target.reshape(pred.shape)
    return jnp.mean((pred - target)**2)


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
