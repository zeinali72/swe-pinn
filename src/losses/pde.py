"""Core PDE residual losses for the Shallow Water Equations."""
import jax
import jax.numpy as jnp
import jax.nn
from flax import linen as nn
from flax.core import FrozenDict
from typing import Dict, Any, Optional

from src.physics import SWEPhysics
from src.data import bathymetry_fn


def compute_pde_loss(model: nn.Module, params: Dict[str, Any], pde_batch: jnp.ndarray,
                     config: FrozenDict, pde_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Compute the PDE residual mean squared error (MSE) for the SWE."""
    if pde_batch.shape[-1] != 3:
        raise ValueError(f"PDE batch requires shape (N, 3), but got {pde_batch.shape}")
    if pde_mask is None:
        pde_mask = jnp.ones((pde_batch.shape[0],), dtype=bool)

    def U_fn(pts):
        return model.apply({'params': params['params']}, pts, train=False)

    jac_U = jax.vmap(jax.jacfwd(U_fn))(pde_batch)
    dU_dx, dU_dy, dU_dt = jac_U[..., 0], jac_U[..., 1], jac_U[..., 2]

    # Use the same function (and train flag) for predictions fed to SWEPhysics
    U_pred = U_fn(pde_batch)

    # 1. Unpack coordinates
    x_batch = pde_batch[..., 0]
    y_batch = pde_batch[..., 1]

    # 2. Get Bathymetry Slopes (Differentiable)
    _, bed_grad_x, bed_grad_y = bathymetry_fn(x_batch, y_batch)

    eps = config["numerics"]["eps"]
    physics = SWEPhysics(U_pred, eps=eps)

    g = config["physics"]["g"]
    n_manning = config["physics"]["n_manning"]
    inflow = config["physics"]["inflow"]

    JF, JG = physics.flux_jac(g=g)
    div_F = jnp.einsum('nij,nj->ni', JF, dU_dx)
    div_G = jnp.einsum('nij,nj->ni', JG, dU_dy)

    # 3. Pass gradients to source term
    S = physics.source(g=g, n_manning=n_manning, inflow=inflow,
                       bed_grad_x=bed_grad_x, bed_grad_y=bed_grad_y)

    residual = (dU_dt + div_F + div_G - S)

    h_mask = jnp.where(U_pred[..., 0] < eps, 0.0, 1.0)
    final_residual = residual * h_mask[..., None]* pde_mask[..., None]
    return jnp.mean(final_residual ** 2)

def compute_neg_h_loss(model: nn.Module, params: Dict[str, Any], pde_points: jnp.ndarray,
                     pde_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Compute penalty for negative water height."""
    U_pred = model.apply({'params': params['params']}, pde_points, train=False)
    h_pred = U_pred[..., 0]
    if pde_mask is not None:
        h_pred = h_pred * pde_mask
    return jnp.mean(jax.nn.relu(-h_pred) ** 2)

def compute_ic_loss(model: nn.Module, params: Dict[str, Any], ic_batch: jnp.ndarray) -> jnp.ndarray:
    """Compute initial condition loss for h=0, hu=0, hv=0 at t=0."""
    U_pred = model.apply({'params': params['params']}, ic_batch, train=False)
    return jnp.mean(U_pred**2)
