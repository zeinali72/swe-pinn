"""Observational data loss for data-driven training."""
import jax.numpy as jnp
from flax import linen as nn
from flax.core import FrozenDict
from typing import Dict, Any


def compute_data_loss(model: nn.Module, params: Dict[str, Any],
                      data_batch: jnp.ndarray, config: FrozenDict) -> jnp.ndarray:
    """Data loss for sparse observations."""
    points_batch = data_batch[:, [1, 2, 0]]  # x, y, t
    h_true = data_batch[:, 3]
    u_true = data_batch[:, 4]
    v_true = data_batch[:, 5]

    U_pred = model.apply({'params': params['params']}, points_batch, train=False)
    h_pred = U_pred[..., 0]
    hu_pred = U_pred[..., 1]
    hv_pred = U_pred[..., 2]

    eps = config["numerics"]["eps"]
    h_true_safe = jnp.maximum(h_true, eps)
    hu_true = h_true_safe * u_true
    hv_true = h_true_safe * v_true

    return (jnp.mean((h_pred - h_true)**2) +
            jnp.mean((hu_pred - hu_true)**2) +
            jnp.mean((hv_pred - hv_true)**2))
