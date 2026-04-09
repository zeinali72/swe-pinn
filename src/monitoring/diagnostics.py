"""Training diagnostics: negative depth stats."""
import jax.numpy as jnp
from typing import Dict


def compute_negative_depth_diagnostics(
    model, params: dict, pde_points: jnp.ndarray,
) -> Dict[str, float]:
    """Compute negative depth diagnostics on a batch of PDE points.

    Returns a dict with keys matching the spec (B.2):
        count    -- number of points where h < 0
        fraction -- fraction of points with h < 0
        min      -- most negative predicted depth (m)
        mean     -- mean of negative predictions only (m)
    """
    if pde_points is None or pde_points.shape[0] == 0:
        return {'count': 0, 'fraction': 0.0, 'min': 0.0, 'mean': 0.0}

    U_pred = model.apply({'params': params['params']}, pde_points, train=False)
    h_pred = U_pred[..., 0]

    # Use a small tolerance so IEEE -0.0 and tiny float-noise are not
    # counted as genuine negative predictions.
    _NEG_TOL = 1e-10
    neg_mask = h_pred < -_NEG_TOL
    n_neg = int(jnp.sum(neg_mask))
    n_total = h_pred.shape[0]

    if n_neg == 0:
        return {'count': 0, 'fraction': 0.0, 'min': 0.0, 'mean': 0.0}

    neg_values = h_pred[neg_mask]
    return {
        'count': n_neg,
        'fraction': n_neg / n_total,
        'min': float(jnp.min(neg_values)),
        'mean': float(jnp.mean(neg_values)),
    }

