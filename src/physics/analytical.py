"""Analytical solutions for validation (dam-break).

The 1D dam-break solution on a flat domain with Manning friction:
  h(x,t) = max((7/3) * n^2 * u^2 * (u*t - x), 0)^(3/7)

Since the problem is 1D in x with constant velocity u_const behind the
wave front:
  hu(x,t) = h(x,t) * u_const   (where h > 0, zero elsewhere)
  hv(x,t) = 0                   (no y-velocity in 1D problem)
"""
import jax.numpy as jnp


def h_exact(x: jnp.ndarray, t: jnp.ndarray, n_manning: float, u_const: float) -> jnp.ndarray:
    """Compute the analytical solution for water depth h(x, t)."""
    arg = (7 / 3) * n_manning**2 * u_const**2 * (u_const * t - x)
    return (jnp.maximum(arg, 0)) ** (3 / 7)


def hu_exact(x: jnp.ndarray, t: jnp.ndarray, n_manning: float, u_const: float) -> jnp.ndarray:
    """Compute the analytical specific discharge hu(x, t) = h * u_const.

    Behind the wave front (where h > 0), velocity is constant at u_const.
    Ahead of the wave front (where h = 0), hu = 0.
    """
    return h_exact(x, t, n_manning, u_const) * u_const


def hv_exact(x: jnp.ndarray, t: jnp.ndarray, n_manning: float, u_const: float) -> jnp.ndarray:
    """Compute hv(x, t) = 0 (no y-velocity in 1D dam-break)."""
    return jnp.zeros_like(x)
