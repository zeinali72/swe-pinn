"""Boundary and initial-condition accuracy metrics: C1-C4."""
import jax
import jax.numpy as jnp
import numpy as np

from src.metrics.boundary import (
    slip_violation as _slip_violation,
    inflow_accuracy as _inflow_accuracy,
    initial_condition_accuracy as _initial_condition_accuracy,
)


def slip_violation(model, params: dict, wall_points: np.ndarray, normals: np.ndarray, eps: float) -> dict:
    """C1: No-slip (zero normal velocity) violation at wall boundaries."""
    return _slip_violation(model, params, jnp.asarray(wall_points), jnp.asarray(normals), eps)


def inflow_boundary_error(
    model,
    params: dict,
    inflow_coords: np.ndarray,
    h_prescribed: np.ndarray,
    t_coords: np.ndarray,
) -> dict:
    """C2: RMSE of h and hu at the inflow boundary.

    Args:
        inflow_coords: (K, 3) inflow boundary coordinates [x, y, t].
        h_prescribed: (K,) prescribed water depth at inflow.
        t_coords: (K,) time coordinates (unused directly; h_prescribed already evaluated).

    Returns:
        Dict with 'rmse_h', 'rmse_hu'.
    """
    inflow_coords = jnp.asarray(inflow_coords)
    h_prescribed = jnp.asarray(h_prescribed)

    if inflow_coords.shape[0] == 0:
        return {"rmse_h": 0.0, "rmse_hu": 0.0}

    flax_params = {"params": params["params"]}
    U = model.apply(flax_params, inflow_coords, train=False)
    h_pred = U[..., 0]
    hu_pred = U[..., 1]

    rmse_h = float(jnp.sqrt(jnp.mean((h_pred - h_prescribed) ** 2)))
    rmse_hu = float(jnp.sqrt(jnp.mean(hu_pred ** 2)))
    return {"rmse_h": rmse_h, "rmse_hu": rmse_hu}


def outflow_gradient_residual(model, params: dict, outflow_coords: np.ndarray, config: dict) -> dict:
    """C3: Outflow zero-gradient residual |dh/dx| at outflow boundary (Exp 1 only).

    Uses jax.vmap(jax.jacfwd(...)) to compute dU/dx at outflow points.
    For Exp 1, outflow is at x=lx, normal n=(1,0,0), so gradient = dh/dx.

    Returns:
        Dict with 'mean_abs', 'max_abs', 'std'.
    """
    outflow_coords = jnp.asarray(outflow_coords)

    if outflow_coords.shape[0] == 0:
        return {"mean_abs": 0.0, "max_abs": 0.0, "std": 0.0}

    flax_params = {"params": params["params"]}

    def U_fn(pts):
        return model.apply(flax_params, pts, train=False)

    # jac shape: (N, 3_outputs, 3_inputs)
    jac_U = jax.vmap(jax.jacfwd(U_fn))(outflow_coords)
    dh_dx = jnp.abs(jac_U[:, 0, 0])  # dh/dx at each outflow point

    return {
        "mean_abs": float(jnp.mean(dh_dx)),
        "max_abs": float(jnp.max(dh_dx)),
        "std": float(jnp.std(dh_dx)),
    }


def initial_condition_error(model, params: dict, ic_coords: np.ndarray) -> dict:
    """C4: IC accuracy — RMSE and max absolute error at t=0."""
    return _initial_condition_accuracy(model, params, jnp.asarray(ic_coords))
