"""Analytical solutions for validation (dam-break)."""
import jax.numpy as jnp


def h_exact(x: jnp.ndarray, t: jnp.ndarray, n_manning: float, u_const: float) -> jnp.ndarray:
    """Compute the analytical solution for water depth h(x, t)."""
    arg = (7 / 3) * n_manning**2 * u_const**2 * (u_const * t - x)
    return (jnp.maximum(arg, 0)) ** (3 / 7)
