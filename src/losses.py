# src/losses.py
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict
from typing import Dict, Any

from src.physics import SWEPhysics, h_exact

def compute_pde_loss(model: nn.Module, params: Dict[str, Any], pde_batch: jnp.ndarray, config: FrozenDict) -> jnp.ndarray:
    """Compute the PDE residual mean squared error (MSE) for the SWE."""
    U_pred = model.apply({'params': params['params']}, pde_batch, train=True)
    def U_fn(pts):
        return model.apply({'params': params['params']}, pts, train=False)

    jac_U = jax.vmap(jax.jacfwd(U_fn))(pde_batch)
    dU_dx, dU_dy, dU_dt = jac_U[..., 0], jac_U[..., 1], jac_U[..., 2]

    # --- FIX: Get eps from config and pass it to SWEPhysics ---
    eps = config["numerics"]["eps"]
    physics = SWEPhysics(U_pred, eps=eps)

    g = config["physics"]["g"]
    n_manning = config["physics"]["n_manning"]
    inflow = config["physics"]["inflow"]

    JF, JG = physics.flux_jac(g=g)
    div_F = jnp.einsum('nij,nj->ni', JF, dU_dx)
    div_G = jnp.einsum('nij,nj->ni', JG, dU_dy)
    S = physics.source(g=g, n_manning=n_manning, inflow=inflow)

    h_mask = jnp.where(U_pred[..., 0] < eps, 0.0, 1.0)
    residual = (dU_dt + div_F + div_G - S) * h_mask[..., None]
    return jnp.mean(residual ** 2)

def compute_ic_loss(model: nn.Module, params: Dict[str, Any], ic_batch: jnp.ndarray) -> jnp.ndarray:
    """Compute initial condition loss for h=0, hu=0, hv=0 at t=0."""
    U_pred = model.apply({'params': params['params']}, ic_batch, train=False)
    err = U_pred[..., 0]**2 + U_pred[..., 1]**2 + U_pred[..., 2]**2
    return jnp.mean(err)

def compute_bc_loss(model: nn.Module, params: Dict[str, Any],
                    left_batch: jnp.ndarray, right_batch: jnp.ndarray,
                    bottom_batch: jnp.ndarray, top_batch: jnp.ndarray,
                    config: FrozenDict) -> jnp.ndarray:
    """Compute boundary condition loss for all domain boundaries."""
    u_const = config["physics"]["u_const"]
    n_manning = config["physics"]["n_manning"]

    U_left = model.apply({'params': params['params']}, left_batch, train=False)
    h_pred_left, hu_pred_left = U_left[..., 0], U_left[..., 1]
    t_left = left_batch[..., 2]
    h_true_left = h_exact(0.0, t_left, n_manning, u_const)
    hu_true_left = h_true_left * u_const
    res_left_h = h_pred_left - h_true_left
    res_left_hu = hu_pred_left - hu_true_left

    def U_fn(pts):
        return model.apply({'params': params['params']}, pts, train=False)
    jac_U_right = jax.vmap(jax.jacfwd(U_fn))(right_batch)
    dU_dx_right = jac_U_right[..., 0]
    res_right_grad = dU_dx_right

    U_bottom = model.apply({'params': params['params']}, bottom_batch, train=False)
    res_bottom_hv = U_bottom[..., 2]
    U_top = model.apply({'params': params['params']}, top_batch, train=False)
    res_top_hv = U_top[..., 2]

    loss = (jnp.mean(res_left_h**2) + jnp.mean(res_left_hu**2) +
            jnp.mean(res_right_grad**2) +
            jnp.mean(res_bottom_hv**2) + jnp.mean(res_top_hv**2))
    return loss

def compute_building_bc_loss(model: nn.Module, params: Dict[str, Any],
                             building_left_batch: jnp.ndarray,
                             building_right_batch: jnp.ndarray,
                             building_bottom_batch: jnp.ndarray,
                             building_top_batch: jnp.ndarray) -> jnp.ndarray:
    """
    Compute slip boundary condition loss for a rectangular building obstacle.
    """
    # Left wall (x-momentum is zero)
    U_left = model.apply({'params': params['params']}, building_left_batch, train=False)
    loss_left = jnp.mean(U_left[..., 1]**2)  # hu**2

    # Right wall (x-momentum is zero)
    U_right = model.apply({'params': params['params']}, building_right_batch, train=False)
    loss_right = jnp.mean(U_right[..., 1]**2)  # hu**2

    # Bottom wall (y-momentum is zero)
    U_bottom = model.apply({'params': params['params']}, building_bottom_batch, train=False)
    loss_bottom = jnp.mean(U_bottom[..., 2]**2)  # hv**2

    # Top wall (y-momentum is zero)
    U_top = model.apply({'params': params['params']}, building_top_batch, train=False)
    loss_top = jnp.mean(U_top[..., 2]**2)  # hv**2

    return loss_left + loss_right + loss_bottom + loss_top

def total_loss(terms: Dict[str, jnp.ndarray], weights: Dict[str, float]) -> jnp.ndarray:
    """Compute the weighted sum of loss terms."""
    loss = (weights.get('pde', 0.0) * terms.get('pde', 0.0) +
            weights.get('ic', 0.0) * terms.get('ic', 0.0) +
            weights.get('bc', 0.0) * terms.get('bc', 0.0))

    # Conditionally add building loss if it exists in both terms and weights
    if 'building_bc' in terms and 'building_bc' in weights:
        loss += weights['building_bc'] * terms['building_bc']

    return loss