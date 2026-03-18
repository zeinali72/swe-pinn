"""ReLoBRaLo — Relative Loss Balancing Residual-based Loss algorithm for PINNs.

Designed specifically for Physics-Informed Neural Networks where loss terms
span many orders of magnitude and some constraints can reach zero and stay there.

Algorithm (per epoch, after warmup)
-------------------------------------
1. Convert MSE → L2:  L_i(t) = sqrt(MSE_i(t))
2. Relative progress: r_i(t) = L_i(t) / (L_i(t_ref) + eps)
   where L_i(t_ref) is the L2 norm at the end of the warmup period.
3. Mean relative:     r̄(t)  = mean_j( r_j(t) )
4. Raw weight:        ρ_i(t) = r̄(t) / (r_i(t) + eps)
   Terms that have improved least get highest weight.
5. EMA smoothing:     λ_i(t) = α·λ_i(t-1) + (1-α)·ρ_i(t)
6. Floor:             λ_i(t) = max(λ_i(t), min_weight)
   Fully-satisfied constraints (L_i ≤ eps) are pinned to min_weight.

Reference: Bischof & Kraus, "Multi-Objective Loss Balancing for Physics-Informed
Deep Learning", arXiv:2110.09813.

Usage
-----
::

    relobralo = ReLoBRaLo(loss_keys=['pde', 'ic', 'bc', 'neg_h'], alpha=0.999, warmup=20)
    scan_body = make_scan_body_relobralo(
        model, optimiser, tuple(relobralo.loss_keys), cfg, data_free, compute_losses
    )
    carry = (params, opt_state, relobralo.weights_array())

    for epoch in range(epochs):
        carry, (terms_stacked, total_stacked) = lax.scan(scan_body, carry, scan_inputs)
        avg_losses = {k: float(terms_stacked[k].sum()) / num_batches for k in terms_stacked}
        relobralo.update(avg_losses)
        carry = (carry[0], carry[1], relobralo.weights_array())
"""
import functools
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax


class ReLoBRaLo:
    """Relative Loss Balancing for PINNs.

    Parameters
    ----------
    loss_keys : list of str
        Ordered list of active loss term names.
    alpha : float
        EMA decay for weight smoothing.  ``alpha=0.999`` gives a very stable
        (slow) adaptation; ``alpha=0.9`` adapts faster.  Defaults to ``0.999``.
    eps : float
        Numerical stability constant.
    warmup : int
        Epochs to hold uniform weights.  At the end of warmup the current L2
        norms are recorded as the reference losses ``L_i(t_ref)``.
    min_weight : float
        Weight floor applied after EMA.  Fully-satisfied constraints
        (``L_i ≤ eps``) are pinned to this value regardless of EMA.
    """

    def __init__(
        self,
        loss_keys: List[str],
        alpha: float = 0.999,
        eps: float = 1e-8,
        warmup: int = 20,
        min_weight: float = 0.01,
    ):
        self.loss_keys = list(loss_keys)
        self.alpha = alpha
        self.eps = eps
        self.warmup = warmup
        self.min_weight = min_weight
        self._epoch = 0
        n = len(loss_keys)
        self._ref_l2: Optional[Dict[str, float]] = None   # set at end of warmup
        self._last_l2: Dict[str, float] = {k: 1.0 for k in loss_keys}
        self._weights: Dict[str, float] = {k: 1.0 / n for k in loss_keys}

    @staticmethod
    def _to_l2(losses: Dict[str, float]) -> Dict[str, float]:
        """MSE → L2 norm: ``sqrt(MSE)``."""
        return {k: float(v ** 0.5) for k, v in losses.items()}

    def update(self, current_losses: Dict[str, float]) -> Dict[str, float]:
        """Update weights from current epoch's MSE losses.

        Parameters
        ----------
        current_losses : dict
            Mapping from loss key to its **MSE** value for this epoch.

        Returns
        -------
        dict
            Updated weight for each loss key.
        """
        self._epoch += 1
        current_l2 = self._to_l2(current_losses)

        if self._epoch <= self.warmup:
            # Accumulate L2 norms; the last warmup epoch becomes the reference.
            self._last_l2 = {
                k: current_l2.get(k, self._last_l2.get(k, 1.0))
                for k in self.loss_keys
            }
            return dict(self._weights)

        # First post-warmup call: fix reference losses.
        if self._ref_l2 is None:
            self._ref_l2 = dict(self._last_l2)
            print(f"  ReLoBRaLo: reference L2 norms set at epoch {self._epoch}: "
                  + ", ".join(f"{k}={v:.3e}" for k, v in self._ref_l2.items()))

        # Step 2: relative progress r_i = L_i(t) / (L_i(ref) + eps)
        rel = {
            k: current_l2.get(k, 0.0) / (self._ref_l2.get(k, 1.0) + self.eps)
            for k in self.loss_keys
        }

        # Step 3: mean relative progress
        r_mean = sum(rel.values()) / max(len(rel), 1)

        # Step 4: raw weights — higher weight for terms that improved less
        rho = {k: r_mean / (rel[k] + self.eps) for k in self.loss_keys}

        # Step 5: EMA smoothing + floor, with zero-loss override
        new_weights = {}
        for k in self.loss_keys:
            if current_l2.get(k, 0.0) <= self.eps:
                # Constraint fully satisfied — pin to min_weight
                new_weights[k] = self.min_weight
            else:
                ema = self.alpha * self._weights[k] + (1.0 - self.alpha) * rho[k]
                new_weights[k] = max(ema, self.min_weight)

        self._last_l2 = {
            k: current_l2.get(k, self._last_l2.get(k, 0.0))
            for k in self.loss_keys
        }
        self._weights = new_weights
        return dict(self._weights)

    @property
    def weights(self) -> Dict[str, float]:
        """Current weights as a plain dict."""
        return dict(self._weights)

    def weights_array(self) -> jnp.ndarray:
        """Current weights as a JAX array ordered by ``self.loss_keys``."""
        return jnp.array([self._weights[k] for k in self.loss_keys], dtype=jnp.float32)


# ---------------------------------------------------------------------------
# JAX training step for adaptive (ReLoBRaLo / any dynamic) weighting
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=[
        'model', 'optimiser', 'config',
        'compute_losses_fn', 'data_free', 'loss_keys',
    ],
)
def train_step_adaptive(
    model,
    optimiser,
    params,
    opt_state,
    batch,
    config,
    data_free: bool,
    compute_losses_fn,
    loss_keys: Tuple[str, ...],
    loss_weights: jnp.ndarray,
):
    """One gradient-descent step with dynamically-supplied loss weights.

    Parameters
    ----------
    loss_keys : tuple of str
        Static, ordered list of active loss term names.
    loss_weights : jnp.ndarray, shape ``(len(loss_keys),)``
        Current weights as a **traced** JAX array — no retracing when weights
        change between epochs.
    """

    def loss_fn(p):
        terms = compute_losses_fn(model, p, batch, config, data_free)
        terms_arr = jnp.stack([terms.get(k, jnp.zeros(())) for k in loss_keys])
        total = jnp.dot(loss_weights, terms_arr)
        return total, terms

    (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimiser.update(grads, opt_state, params, value=loss_val)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, metrics, loss_val


def make_scan_body_relobralo(
    model,
    optimiser,
    loss_keys: Tuple[str, ...],
    config,
    data_free: bool,
    compute_losses_fn,
):
    """Return a ``scan_body`` for ``lax.scan`` with adaptive weight carry.

    **Carry format:** ``(params, opt_state, loss_weights)``

    ``loss_weights`` is a traced JAX array that flows through every batch
    within an epoch unchanged and is updated between epochs by
    :meth:`ReLoBRaLo.update`.
    """

    def scan_body(carry, batch_data):
        params, opt_state, loss_weights = carry
        new_params, new_opt_state, terms, total = train_step_adaptive(
            model, optimiser, params, opt_state,
            batch_data, config, data_free, compute_losses_fn,
            loss_keys, loss_weights,
        )
        return (new_params, new_opt_state, loss_weights), (terms, total)

    return scan_body
