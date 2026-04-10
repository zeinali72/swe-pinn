"""Importance sampling for PINN collocation points (arXiv:2104.12325).

Implements Algorithm 2 from:
  "Efficient Training of Physics-Informed Neural Networks via Importance Sampling"
  Wu et al., 2021.

The key idea: concentrate collocation points in high-residual regions.
  1. Maintain a large GPU-resident pool of N candidate PDE points.
  2. Every ``resample_freq`` epochs, evaluate PDE residuals on the full pool
     in a single JIT call (XLA-chunked via ``lax.map`` to bound memory).
  3. Each epoch, draw a fresh n_active sample from the pool using the current
     probability distribution.
  4. Correct the gradient estimator with importance weights w_i = 1/(N*p_i).
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random, lax
from flax.core import FrozenDict

from src.physics import SWEPhysics
from src.data.bathymetry import bathymetry_fn
from src.config import get_dtype


# ---------------------------------------------------------------------------
# Per-point residuals (building block — works on a single chunk)
# ---------------------------------------------------------------------------

def pde_residuals_per_point(
    model,
    params,
    pde_batch: jnp.ndarray,
    config: FrozenDict,
) -> jnp.ndarray:
    """Compute per-point squared SWE PDE residuals.

    Parameters
    ----------
    pde_batch : jnp.ndarray, shape (N, 3)
        Collocation points (x, y, t).

    Returns
    -------
    jnp.ndarray, shape (N,)
        Sum of squared residuals across the 3 SWE equations per point.
    """
    def U_fn(pts):
        return model.apply({'params': params['params']}, pts, train=False)

    U_pred = U_fn(pde_batch)
    jac_U = jax.vmap(jax.jacfwd(U_fn))(pde_batch)
    dU_dx, dU_dy, dU_dt = jac_U[..., 0], jac_U[..., 1], jac_U[..., 2]

    x_batch = pde_batch[..., 0]
    y_batch = pde_batch[..., 1]
    _, bed_grad_x, bed_grad_y = bathymetry_fn(x_batch, y_batch)

    eps = config["numerics"]["eps"]
    physics = SWEPhysics(U_pred, eps=eps)

    g = config["physics"]["g"]
    n_manning = config["physics"]["n_manning"]
    inflow = config["physics"]["inflow"]

    JF, JG = physics.flux_jac(g=g)
    div_F = jnp.einsum('nij,nj->ni', JF, dU_dx)
    div_G = jnp.einsum('nij,nj->ni', JG, dU_dy)
    Cf = config.get("physics", {}).get("Cf", None)
    S = physics.source(g=g, n_manning=n_manning, inflow=inflow,
                       bed_grad_x=bed_grad_x, bed_grad_y=bed_grad_y, Cf=Cf)

    residual = dU_dt + div_F + div_G - S
    h_mask = jnp.where(U_pred[..., 0] < eps, 0.0, 1.0)
    masked = residual * h_mask[..., None]

    return jnp.sum(masked ** 2, axis=-1)


# ---------------------------------------------------------------------------
# All-GPU pool evaluation (single JIT, XLA-chunked)
# ---------------------------------------------------------------------------

def evaluate_pool_residuals(
    model,
    params,
    pool_gpu: jnp.ndarray,
    config: FrozenDict,
    chunk_size: int,
) -> jnp.ndarray:
    """Evaluate PDE residuals on the full pool in a single JIT call.

    Uses ``lax.map`` to process fixed-size chunks sequentially inside XLA,
    bounding peak memory to one chunk while keeping the entire computation
    on GPU with no Python-level loop or CPU round-trips.

    Parameters
    ----------
    pool_gpu : jnp.ndarray, shape (N, 3)
        Full collocation pool resident on GPU.
    chunk_size : int
        Number of points per XLA chunk (controls memory vs. speed).

    Returns
    -------
    jnp.ndarray, shape (N,)
    """
    n = pool_gpu.shape[0]
    # Pad to exact multiple of chunk_size (trace-time constant)
    n_padded = ((n + chunk_size - 1) // chunk_size) * chunk_size
    padded = jnp.pad(pool_gpu, ((0, n_padded - n), (0, 0)))
    chunks = padded.reshape(-1, chunk_size, 3)

    def _eval_chunk(chunk):
        return pde_residuals_per_point(model, params, chunk, config)

    all_res = lax.map(_eval_chunk, chunks)      # (n_chunks, chunk_size)
    return all_res.reshape(-1)[:n]


# ---------------------------------------------------------------------------
# GPU-side probability computation
# ---------------------------------------------------------------------------

def compute_sampling_probs(
    residuals: jnp.ndarray,
    alpha: float,
) -> jnp.ndarray:
    """Compute importance sampling probabilities on GPU.

    p_j = alpha * (r_j / sum(r)) + (1 - alpha) / N

    Parameters
    ----------
    residuals : jnp.ndarray, shape (N,)
    alpha : float
        Mixture coefficient (1.0 = pure error, 0.0 = uniform).

    Returns
    -------
    jnp.ndarray, shape (N,)
        Normalised probability distribution.
    """
    n = residuals.shape[0]
    err_sum = jnp.sum(residuals)
    # When all residuals are zero, fall back to uniform
    p_error = jnp.where(err_sum > 1e-12, residuals / err_sum, 1.0 / n)
    probs = alpha * p_error + (1.0 - alpha) / n
    probs = jnp.maximum(probs, 0.0)
    return probs / jnp.sum(probs)


# ---------------------------------------------------------------------------
# GPU-side sampling + weight computation
# ---------------------------------------------------------------------------

def sample_from_pool(
    key: jnp.ndarray,
    pool_gpu: jnp.ndarray,
    probs_gpu: jnp.ndarray,
    n_active: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Draw n_active points from pool using current probabilities (all on GPU).

    Uses Gumbel-top-k for efficient without-replacement sampling.

    Parameters
    ----------
    pool_gpu : jnp.ndarray, shape (N, 3)
    probs_gpu : jnp.ndarray, shape (N,)
    n_active : int

    Returns
    -------
    selected_pts : jnp.ndarray, shape (n_active, 3)
    importance_weights : jnp.ndarray, shape (n_active,), mean ≈ 1.0
    """
    pool_size = pool_gpu.shape[0]
    indices = random.choice(key, pool_size, shape=(n_active,), p=probs_gpu, replace=False)
    selected_pts = pool_gpu[indices]

    # Importance correction: w_i = 1 / (N * p_i), normalised to mean = 1
    selected_probs = probs_gpu[indices]
    weights_unnorm = 1.0 / (pool_size * selected_probs)
    importance_weights = weights_unnorm / jnp.mean(weights_unnorm)

    return selected_pts, importance_weights


# ---------------------------------------------------------------------------
# Weighted PDE loss (used inside the training step)
# ---------------------------------------------------------------------------

def compute_weighted_pde_loss(
    model,
    params,
    pde_batch: jnp.ndarray,
    weights: jnp.ndarray,
    config: FrozenDict,
) -> jnp.ndarray:
    """Importance-corrected PDE loss: mean(residuals * weights).

    Produces an unbiased gradient estimate (Equation 16 in arXiv:2104.12325).
    """
    per_point = pde_residuals_per_point(model, params, pde_batch, config)
    return jnp.mean(per_point * weights)
