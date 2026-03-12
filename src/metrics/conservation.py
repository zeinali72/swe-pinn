"""Conservation metrics: volume balance and continuity residual via autodiff."""
import jax
import jax.numpy as jnp

from src.physics import SWEPhysics
from src.data import bathymetry_fn


def volume_balance(
    pred_h: jnp.ndarray,
    coords: jnp.ndarray,
    domain_bounds: dict,
    n_time_steps: int = 20,
) -> dict:
    """Estimate volume balance over the spatial domain at discrete time steps.

    Uses trapezoidal-style averaging: for each time bin, the mean predicted
    depth is multiplied by the domain area to give an approximate total volume.

    Args:
        pred_h: (N,) predicted water depth.
        coords: (N, 3) coordinates [x, y, t].
        domain_bounds: Dict with ``lx``, ``ly``, ``t_final`` (and optional ``x_min``, ``y_min``).
        n_time_steps: Number of time bins.

    Returns:
        Dict with ``max_mass_error_pct``, ``final_mass_error_pct``,
        ``volume_time_series`` (list of (t, volume) tuples).
    """
    t = coords[:, 2]
    t_final = domain_bounds.get("t_final", float(jnp.max(t)))
    lx = domain_bounds["lx"]
    ly = domain_bounds["ly"]
    area = lx * ly

    t_edges = jnp.linspace(0.0, t_final, n_time_steps + 1)
    volumes = []
    for i in range(n_time_steps):
        mask = (t >= t_edges[i]) & (t < t_edges[i + 1])
        count = jnp.sum(mask)
        if count > 0:
            mean_h = jnp.mean(jnp.where(mask, pred_h, 0.0)) * pred_h.shape[0] / count
            volumes.append((float((t_edges[i] + t_edges[i + 1]) / 2), float(mean_h * area)))
        else:
            volumes.append((float((t_edges[i] + t_edges[i + 1]) / 2), 0.0))

    if len(volumes) < 2:
        return {
            "max_mass_error_pct": 0.0,
            "final_mass_error_pct": 0.0,
            "volume_time_series": volumes,
        }

    vol_vals = [v for _, v in volumes]
    initial_vol = vol_vals[0] if vol_vals[0] > 0 else 1.0
    errors = [abs(v - vol_vals[0]) / max(initial_vol, 1e-12) * 100 for v in vol_vals]

    return {
        "max_mass_error_pct": max(errors),
        "final_mass_error_pct": errors[-1],
        "volume_time_series": volumes,
    }


def continuity_residual(
    model,
    params: dict,
    sample_coords: jnp.ndarray,
    config: dict,
) -> dict:
    """Compute the mass-conservation (continuity) residual via autodiff.

    Residual: R_mass = dh/dt + d(hu)/dx + d(hv)/dy

    This reuses the same Jacobian pattern as ``src/losses/pde.py``.

    Args:
        model: Flax model.
        params: Parameter dict (must contain ``params`` key).
        sample_coords: (N, 3) points at which to evaluate the residual.
        config: Full config dict (needs ``numerics.eps``, ``physics.*``).

    Returns:
        Dict with ``mean_abs``, ``max_abs``, ``std``.
    """
    flax_params = {"params": params["params"]}

    def U_fn(pts):
        return model.apply(flax_params, pts, train=False)

    jac_U = jax.vmap(jax.jacfwd(U_fn))(sample_coords)
    dU_dx = jac_U[..., 0]  # (N, 3) derivatives w.r.t. x
    dU_dy = jac_U[..., 1]
    dU_dt = jac_U[..., 2]

    # Continuity equation: dh/dt + d(hu)/dx + d(hv)/dy = source_mass
    dh_dt = dU_dt[..., 0]
    dhu_dx = dU_dx[..., 1]
    dhv_dy = dU_dy[..., 2]

    # Source mass term (inflow)
    inflow = config.get("physics", {}).get("inflow")
    R_mass_source = 0.0 if inflow is None else inflow

    residual = dh_dt + dhu_dx + dhv_dy - R_mass_source

    abs_res = jnp.abs(residual)
    return {
        "mean_abs": float(jnp.mean(abs_res)),
        "max_abs": float(jnp.max(abs_res)),
        "std": float(jnp.std(residual)),
    }
