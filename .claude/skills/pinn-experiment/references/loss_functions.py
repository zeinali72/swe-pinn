"""
Template for composite SWE loss functions used in PINN training.

This module shows the canonical structure for computing PDE residuals
and assembling the composite loss. Adapt for each experiment's specific
boundary conditions and source terms.

Depends on: jax, jax.numpy, src.physics.SWEPhysics
"""

import jax
import jax.numpy as jnp
from typing import Callable, Dict, Tuple


# =============================================================================
# SWE PDE Residuals
# =============================================================================

def swe_residuals(
    predict_fn: Callable,
    params: dict,
    points: jnp.ndarray,
    g: float = 9.81,
    n_manning: float = 0.05,
    eps: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute SWE residuals at collocation points using automatic differentiation.

    The conservative-form 2D SWE:
        dh/dt   + d(hu)/dx + d(hv)/dy = 0                  (continuity)
        d(hu)/dt + d(hu²/h + gh²/2)/dx + d(huv/h)/dy = S_x (x-momentum)
        d(hv)/dt + d(huv/h)/dx + d(hv²/h + gh²/2)/dy = S_y (y-momentum)

    Parameters
    ----------
    predict_fn : callable
        Model forward pass: (params, x, y, t) -> (h, hu, hv).
    params : dict
        Flax model parameters.
    points : jnp.ndarray, shape (N, 3)
        Collocation points (x, y, t).
    g : float
        Gravitational acceleration.
    n_manning : float
        Manning's roughness coefficient.
    eps : float
        Small value for numerical stability in wet/dry transitions.

    Returns
    -------
    r_cont, r_xmom, r_ymom : jnp.ndarray
        Residuals for each equation, shape (N,).
    """
    x, y, t = points[:, 0], points[:, 1], points[:, 2]

    def single_point_residual(x_i, y_i, t_i):
        # Forward pass
        def predict(x_, y_, t_):
            return predict_fn(params, x_, y_, t_)

        h, hu, hv = predict(x_i, y_i, t_i)

        # Ensure h >= eps for stability
        h_safe = jnp.maximum(h, eps)

        # Velocities
        u = hu / h_safe
        v = hv / h_safe

        # Time derivatives
        dh_dt = jax.grad(lambda t_: predict(x_i, y_i, t_)[0])(t_i)
        dhu_dt = jax.grad(lambda t_: predict(x_i, y_i, t_)[1])(t_i)
        dhv_dt = jax.grad(lambda t_: predict(x_i, y_i, t_)[2])(t_i)

        # Spatial derivatives
        dhu_dx = jax.grad(lambda x_: predict(x_, y_i, t_i)[1])(x_i)
        dhv_dy = jax.grad(lambda y_: predict(x_i, y_, t_i)[2])(y_i)

        # Flux derivatives for momentum equations
        # x-momentum: d(hu²/h + gh²/2)/dx
        def flux_x_mom(x_):
            h_, hu_, _ = predict(x_, y_i, t_i)
            h_s = jnp.maximum(h_, eps)
            return hu_**2 / h_s + 0.5 * g * h_**2

        d_flux_x_dx = jax.grad(flux_x_mom)(x_i)

        # Cross-flux: d(hu*hv/h)/dy for x-momentum
        def cross_flux_x(y_):
            h_, hu_, hv_ = predict(x_i, y_, t_i)
            h_s = jnp.maximum(h_, eps)
            return hu_ * hv_ / h_s

        d_cross_x_dy = jax.grad(cross_flux_x)(y_i)

        # y-momentum: d(hv²/h + gh²/2)/dy
        def flux_y_mom(y_):
            h_, _, hv_ = predict(x_i, y_, t_i)
            h_s = jnp.maximum(h_, eps)
            return hv_**2 / h_s + 0.5 * g * h_**2

        d_flux_y_dy = jax.grad(flux_y_mom)(y_i)

        # Cross-flux: d(hu*hv/h)/dx for y-momentum
        def cross_flux_y(x_):
            h_, hu_, hv_ = predict(x_, y_i, t_i)
            h_s = jnp.maximum(h_, eps)
            return hu_ * hv_ / h_s

        d_cross_y_dx = jax.grad(cross_flux_y)(x_i)

        # Manning friction source terms
        vel_mag = jnp.sqrt(u**2 + v**2 + eps)
        S_fx = -g * n_manning**2 * u * vel_mag / (h_safe ** (1.0 / 3.0))
        S_fy = -g * n_manning**2 * v * vel_mag / (h_safe ** (1.0 / 3.0))

        # Residuals
        r_cont = dh_dt + dhu_dx + dhv_dy
        r_xmom = dhu_dt + d_flux_x_dx + d_cross_x_dy - S_fx
        r_ymom = dhv_dt + d_flux_y_dy + d_cross_y_dx - S_fy

        return r_cont, r_xmom, r_ymom

    # Vectorise over all points
    r_cont, r_xmom, r_ymom = jax.vmap(single_point_residual)(x, y, t)

    return r_cont, r_xmom, r_ymom


# =============================================================================
# Individual Loss Terms
# =============================================================================

def pde_loss(
    predict_fn: Callable,
    params: dict,
    pde_points: jnp.ndarray,
    g: float = 9.81,
    n_manning: float = 0.05,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """MSE of SWE residuals at PDE collocation points."""
    r_cont, r_xmom, r_ymom = swe_residuals(
        predict_fn, params, pde_points, g, n_manning, eps
    )
    return jnp.mean(r_cont**2 + r_xmom**2 + r_ymom**2)


def ic_loss(
    predict_fn: Callable,
    params: dict,
    ic_points: jnp.ndarray,
    h_init: jnp.ndarray,
    hu_init: jnp.ndarray,
    hv_init: jnp.ndarray,
) -> jnp.ndarray:
    """MSE of initial condition violations.

    For dry-bed scenarios: h_init=0, hu_init=0, hv_init=0.
    For dam-break: h_init varies spatially.
    """
    x, y, t = ic_points[:, 0], ic_points[:, 1], ic_points[:, 2]
    h_pred, hu_pred, hv_pred = jax.vmap(
        lambda xi, yi, ti: predict_fn(params, xi, yi, ti)
    )(x, y, t)

    return jnp.mean(
        (h_pred - h_init) ** 2
        + (hu_pred - hu_init) ** 2
        + (hv_pred - hv_init) ** 2
    )


def bc_loss_slip(
    predict_fn: Callable,
    params: dict,
    bc_points: jnp.ndarray,
    normals: jnp.ndarray,
) -> jnp.ndarray:
    """MSE of slip boundary condition: u · n = 0 at walls.

    Parameters
    ----------
    bc_points : jnp.ndarray, shape (N, 3)
        Boundary points (x, y, t).
    normals : jnp.ndarray, shape (N, 2)
        Outward unit normals at each boundary point.
    """
    x, y, t = bc_points[:, 0], bc_points[:, 1], bc_points[:, 2]
    h, hu, hv = jax.vmap(
        lambda xi, yi, ti: predict_fn(params, xi, yi, ti)
    )(x, y, t)

    eps = 1e-6
    h_safe = jnp.maximum(h, eps)
    u = hu / h_safe
    v = hv / h_safe

    # u · n = u * nx + v * ny
    u_dot_n = u * normals[:, 0] + v * normals[:, 1]

    return jnp.mean(u_dot_n**2)


def bc_loss_inflow(
    predict_fn: Callable,
    params: dict,
    inflow_points: jnp.ndarray,
    prescribed_hu: jnp.ndarray,
) -> jnp.ndarray:
    """MSE of inflow boundary condition: hu = prescribed discharge."""
    x, y, t = inflow_points[:, 0], inflow_points[:, 1], inflow_points[:, 2]
    _, hu_pred, _ = jax.vmap(
        lambda xi, yi, ti: predict_fn(params, xi, yi, ti)
    )(x, y, t)

    return jnp.mean((hu_pred - prescribed_hu) ** 2)


def data_loss(
    predict_fn: Callable,
    params: dict,
    data_points: jnp.ndarray,
    h_obs: jnp.ndarray,
    hu_obs: jnp.ndarray,
    hv_obs: jnp.ndarray,
) -> jnp.ndarray:
    """MSE against ICM observational data."""
    x, y, t = data_points[:, 0], data_points[:, 1], data_points[:, 2]
    h_pred, hu_pred, hv_pred = jax.vmap(
        lambda xi, yi, ti: predict_fn(params, xi, yi, ti)
    )(x, y, t)

    return jnp.mean(
        (h_pred - h_obs) ** 2
        + (hu_pred - hu_obs) ** 2
        + (hv_pred - hv_obs) ** 2
    )


# =============================================================================
# Composite Loss
# =============================================================================

def composite_loss(
    predict_fn: Callable,
    params: dict,
    batch: Dict[str, jnp.ndarray],
    weights: Dict[str, float],
    physics_params: Dict[str, float],
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute the total weighted loss.

    Parameters
    ----------
    predict_fn : callable
        Model forward pass.
    params : dict
        Model parameters.
    batch : dict
        Keys: 'pde_points', 'ic_points', 'ic_h', 'ic_hu', 'ic_hv',
        'bc_slip_points', 'bc_normals', 'inflow_points', 'inflow_hu',
        'data_points', 'data_h', 'data_hu', 'data_hv'.
    weights : dict
        Keys: 'pde', 'ic', 'bc', 'data'.
    physics_params : dict
        Keys: 'g', 'n_manning', 'eps'.

    Returns
    -------
    total_loss : jnp.ndarray
        Scalar total loss.
    loss_components : dict
        Individual unweighted loss values for logging.
    """
    g = physics_params.get("g", 9.81)
    n_manning = physics_params.get("n_manning", 0.05)
    eps = physics_params.get("eps", 1e-6)

    l_pde = pde_loss(predict_fn, params, batch["pde_points"], g, n_manning, eps)
    l_ic = ic_loss(
        predict_fn, params, batch["ic_points"],
        batch["ic_h"], batch["ic_hu"], batch["ic_hv"],
    )
    l_bc = bc_loss_slip(
        predict_fn, params, batch["bc_slip_points"], batch["bc_normals"],
    )

    # Data loss (zero if no data points provided)
    if "data_points" in batch and batch["data_points"].shape[0] > 0:
        l_data = data_loss(
            predict_fn, params, batch["data_points"],
            batch["data_h"], batch["data_hu"], batch["data_hv"],
        )
    else:
        l_data = jnp.array(0.0)

    # Add inflow BC if present
    l_inflow = jnp.array(0.0)
    if "inflow_points" in batch and batch["inflow_points"].shape[0] > 0:
        l_inflow = bc_loss_inflow(
            predict_fn, params, batch["inflow_points"], batch["inflow_hu"],
        )
        l_bc = l_bc + l_inflow

    total = (
        weights["pde"] * l_pde
        + weights["ic"] * l_ic
        + weights["bc"] * l_bc
        + weights.get("data", 0.0) * l_data
    )

    components = {
        "total": total,
        "pde": l_pde,
        "ic": l_ic,
        "bc": l_bc,
        "data": l_data,
    }

    return total, components
