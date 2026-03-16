"""Composite loss utilities shared across all experiments."""
import jax.numpy as jnp
from typing import Dict


def total_loss(terms: Dict[str, jnp.ndarray], weights: Dict[str, float]) -> jnp.ndarray:
    """Combine weighted loss terms into a single scalar loss."""
    loss = 0.0
    for key in terms.keys():
        if key in weights:
            loss += weights[key] * terms[key]
    return loss
