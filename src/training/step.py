"""Generic training step shared by all experiments.

Each experiment provides a ``compute_losses_fn`` closure that encapsulates its
specific IC/BC logic.  The generic :func:`train_step` handles gradient
computation, optimizer updates, and weighted loss aggregation.
"""
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

from src.losses.composite import total_loss


def train_step(
    model: Any,
    optimiser: optax.GradientTransformation,
    params: FrozenDict,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    config: Dict[str, Any],
    data_free: bool,
    compute_losses_fn: Callable,
    weights_dict: FrozenDict,
) -> Tuple[FrozenDict, optax.OptState, Dict[str, float], float]:
    """Perform one gradient-descent step.

    Parameters
    ----------
    compute_losses_fn : callable
        ``(model, params, batch, config, data_free) -> dict[str, scalar]``
        Returns a dict of named loss terms (e.g. ``'pde'``, ``'ic'``, ``'bc'``,
        ``'data'``).  Only terms present in *weights_dict* contribute to the
        total; missing keys default to ``0.0``.
    """

    def loss_fn(params):
        terms = compute_losses_fn(model, params, batch, config, data_free)
        terms_with_defaults = {k: terms.get(k, 0.0) for k in weights_dict.keys()}
        total = total_loss(terms_with_defaults, weights_dict)
        return total, terms

    (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    grad_norm = optax.global_norm(grads)
    metrics = {**metrics, "_grad_norm": grad_norm}
    updates, new_opt_state = optimiser.update(grads, opt_state, params, value=loss_val)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, metrics, loss_val


train_step_jitted = jax.jit(
    train_step,
    static_argnames=[
        'model', 'optimiser', 'config',
        'compute_losses_fn', 'weights_dict', 'data_free',
    ],
)
