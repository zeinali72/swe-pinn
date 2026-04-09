"""Boundary-condition and initial-condition accuracy metrics: C1-C4."""
import jax
import jax.numpy as jnp


def slip_violation(
    model,
    params: dict,
    wall_points: jnp.ndarray,
    normals: jnp.ndarray,
    eps: float,
) -> dict:
    """Measure no-slip (zero normal velocity) violation at wall boundaries.

    Args:
        model: Flax model.
        params: Parameter dict with ``params`` key.
        wall_points: (K, 3) wall boundary coordinates [x, y, t].
        normals: (K, 2) outward unit normals [nx, ny].
        eps: Numerical epsilon for safe division.

    Returns:
        Dict with ``mean_slip``, ``max_slip``.
    """
    if wall_points.shape[0] == 0:
        return {"mean_slip": 0.0, "max_slip": 0.0}

    flax_params = {"params": params["params"]}
    U = model.apply(flax_params, wall_points, train=False)
    h = U[..., 0]
    hu = U[..., 1]
    hv = U[..., 2]

    h_safe = jnp.maximum(h, eps)
    u = hu / h_safe
    v = hv / h_safe

    v_normal = jnp.abs(u * normals[:, 0] + v * normals[:, 1])
    return {
        "mean_slip": float(jnp.mean(v_normal)),
        "max_slip": float(jnp.max(v_normal)),
    }


def inflow_accuracy(
    model,
    params: dict,
    inflow_coords: jnp.ndarray,
    bc_fn,
    config: dict,
) -> dict:
    """C2: Inflow boundary error via a callable BC function.

    Args:
        model: Flax model.
        params: Parameter dict with ``params`` key.
        inflow_coords: (K, 3) inflow boundary coordinates [x, y, t].
        bc_fn: Callable ``bc_fn(t) -> h_bc`` returning prescribed water level.
        config: Full config dict.

    Returns:
        Dict with ``rmse_h``, ``rmse_hu``.
    """
    if inflow_coords.shape[0] == 0:
        return {"rmse_h": 0.0, "rmse_hu": 0.0}

    flax_params = {"params": params["params"]}
    U = model.apply(flax_params, inflow_coords, train=False)
    h_pred = U[..., 0]
    hu_pred = U[..., 1]

    t = inflow_coords[:, 2]
    h_bc = bc_fn(t)

    rmse_h = float(jnp.sqrt(jnp.mean((h_pred - h_bc) ** 2)))
    rmse_hu = float(jnp.sqrt(jnp.mean(hu_pred ** 2)))

    return {"rmse_h": rmse_h, "rmse_hu": rmse_hu}


def inflow_boundary_error(
    model,
    params: dict,
    inflow_coords: jnp.ndarray,
    h_prescribed: jnp.ndarray,
) -> dict:
    """C2: Inflow boundary error against a pre-evaluated prescribed depth array.

    Use this variant when you have already evaluated h_bc at the inflow points
    (e.g. from an analytical solution).  See ``inflow_accuracy`` for the
    callable-BC variant.

    Args:
        model: Flax model.
        params: Parameter dict with ``params`` key.
        inflow_coords: (K, 3) inflow boundary coordinates [x, y, t].
        h_prescribed: (K,) prescribed water depth at each inflow point.

    Returns:
        Dict with ``rmse_h``, ``rmse_hu``.
    """
    if inflow_coords.shape[0] == 0:
        return {"rmse_h": 0.0, "rmse_hu": 0.0}

    flax_params = {"params": params["params"]}
    U = model.apply(flax_params, inflow_coords, train=False)
    h_pred = U[..., 0]
    hu_pred = U[..., 1]

    rmse_h = float(jnp.sqrt(jnp.mean((h_pred - jnp.asarray(h_prescribed)) ** 2)))
    rmse_hu = float(jnp.sqrt(jnp.mean(hu_pred ** 2)))

    return {"rmse_h": rmse_h, "rmse_hu": rmse_hu}


def outflow_gradient_residual(
    model,
    params: dict,
    outflow_coords: jnp.ndarray,
    config: dict,
) -> dict:
    """C3: Zero-gradient outflow residual |dh/dn| via autodiff (Exp 1 only).

    For Experiment 1 the outflow is at x = lx with outward normal n = (1, 0),
    so the residual is |dh/dx| evaluated at the right boundary.

    Args:
        model: Flax model.
        params: Parameter dict with ``params`` key.
        outflow_coords: (K, 3) outflow boundary coordinates [x, y, t].
        config: Full config dict (used for dtype consistency).

    Returns:
        Dict with ``mean_abs``, ``max_abs``, ``std``.
    """
    outflow_coords = jnp.asarray(outflow_coords)

    if outflow_coords.shape[0] == 0:
        return {"mean_abs": 0.0, "max_abs": 0.0, "std": 0.0}

    flax_params = {"params": params["params"]}

    def single_U(pt):
        """Forward pass for a single point (1, 3) → (3,)."""
        return model.apply(flax_params, pt[None], train=False)[0]

    # vmap over single-point jacfwd: jac shape (K, 3_outputs, 3_inputs)
    jac_U = jax.vmap(jax.jacfwd(single_U))(outflow_coords)
    dh_dx = jnp.abs(jac_U[:, 0, 0])

    return {
        "mean_abs": float(jnp.mean(dh_dx)),
        "max_abs": float(jnp.max(dh_dx)),
        "std": float(jnp.std(dh_dx)),
    }


def initial_condition_accuracy(
    model,
    params: dict,
    ic_coords: jnp.ndarray,
) -> dict:
    """Measure accuracy of predictions at t=0 (IC should be h=0, hu=0, hv=0).

    Args:
        model: Flax model.
        params: Parameter dict with ``params`` key.
        ic_coords: (K, 3) coordinates at t=0.

    Returns:
        Dict with ``rmse``, ``max_abs_error`` for each variable.
    """
    if ic_coords.shape[0] == 0:
        return {"rmse": 0.0, "max_abs_error": 0.0}

    flax_params = {"params": params["params"]}
    U = model.apply(flax_params, ic_coords, train=False)

    rmse_val = float(jnp.sqrt(jnp.mean(U ** 2)))
    max_err = float(jnp.max(jnp.abs(U)))

    return {
        "rmse": rmse_val,
        "max_abs_error": max_err,
        "rmse_h": float(jnp.sqrt(jnp.mean(U[..., 0] ** 2))),
        "rmse_hu": float(jnp.sqrt(jnp.mean(U[..., 1] ** 2))),
        "rmse_hv": float(jnp.sqrt(jnp.mean(U[..., 2] ** 2))),
    }
