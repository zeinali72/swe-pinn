"""InferenceContext: a bundle of everything metric functions need."""
from dataclasses import dataclass, field
from typing import Any, Optional

import jax.numpy as jnp


@dataclass
class InferenceContext:
    """Immutable bundle passed to metric and reporting functions.

    Attributes:
        config: Full experiment config dict.
        model: Flax nn.Module instance.
        params: Parameter dict (must contain ``params`` key).
        predictor: ``Predictor`` instance for batched forward pass.
        val_coords: (N, 3) validation coordinates [x, y, t].
        val_targets: (N, 1) or (N, 3) reference outputs.
        predictions: (N, 3) model predictions [h, hu, hv].
        experiment_name: Logical experiment name.
        domain_bounds: Dict with ``lx``, ``ly``, ``t_final``, etc.
        experiment_meta: Registry entry for the experiment.
        inference_time_seconds: Wall-clock time for the forward pass.
        training_metadata: Optional checkpoint metadata dict.
        domain_sampler: Optional ``IrregularDomainSampler`` (exp 7/8).
        bc_fn: Optional boundary-condition interpolation function.
        checkpoint_name: Name of the checkpoint evaluated (e.g. ``best_nse``).
    """

    config: dict
    model: Any
    params: dict
    predictor: Any
    val_coords: jnp.ndarray
    val_targets: jnp.ndarray
    predictions: jnp.ndarray
    experiment_name: str
    domain_bounds: dict
    experiment_meta: dict
    inference_time_seconds: float = 0.0
    training_metadata: Optional[dict] = None
    domain_sampler: Optional[Any] = None
    bc_fn: Optional[Any] = None
    checkpoint_name: str = "unknown"
