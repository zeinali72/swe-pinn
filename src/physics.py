# src/physics.py
import jax.numpy as jnp
from typing import Tuple
from src.config import config, EPS

# Extract physics parameters from config
N_MANNING = config["physics"]["n_manning"]
U_CONST = config["physics"]["u_const"]
G = config["physics"]["g"]
INFLOW = config["physics"]["inflow"]

def h_exact(x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    """Compute the analytical solution for water depth h(x, t)."""
    arg = (7 / 3) * N_MANNING**2 * U_CONST**2 * (U_CONST * t - x)
    return (jnp.maximum(arg, 0)) ** (3 / 7)

class SWEPhysics:
    """Compute terms for the 2D Shallow Water Equations (SWE)."""
    def __init__(self, U: jnp.ndarray, bed_elevation: jnp.ndarray = None):
        h = U[..., 0]
        self.hu = U[..., 1]
        self.hv = U[..., 2]
        self.h_safe = jnp.maximum(h, EPS)
        self.u = self.hu / self.h_safe
        self.v = self.hv / self.h_safe
        self.bed = bed_elevation

    def flux_jac(self, g: float = G) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute analytical Jacobians of SWE fluxes."""
        F = jnp.stack([
            jnp.stack([jnp.zeros_like(self.h_safe), jnp.ones_like(self.h_safe), jnp.zeros_like(self.h_safe)], axis=-1),
            jnp.stack([-self.u**2 + g * self.h_safe, 2 * self.u, jnp.zeros_like(self.h_safe)], axis=-1),
            jnp.stack([-self.u * self.v, self.v, self.u], axis=-1)
        ], axis=-2)
        G = jnp.stack([
            jnp.stack([jnp.zeros_like(self.h_safe), jnp.zeros_like(self.h_safe), jnp.ones_like(self.h_safe)], axis=-1),
            jnp.stack([-self.u * self.v, self.v, self.u], axis=-1),
            jnp.stack([-self.v**2 + g * self.h_safe, jnp.zeros_like(self.h_safe), 2 * self.v], axis=-1)
        ], axis=-2)
        return F, G

    def source(self, g: float, n_manning: float, inflow: float = INFLOW) -> jnp.ndarray:
        """Compute source terms for SWE."""
        vel = jnp.sqrt(self.u**2 + self.v**2)
        sfx = n_manning**2 * self.u * vel / (self.h_safe**(4 / 3))
        sfy = n_manning**2 * self.v * vel / (self.h_safe**(4 / 3))
        sox = soy = 0.0
        if self.bed is not None:
            sox, soy = -jnp.gradient(self.bed, axis=-2), -jnp.gradient(self.bed, axis=-1)

        R = 0.0 if inflow is None else inflow
        s_mass = R * jnp.ones_like(self.h_safe)
        s_mom_x = -g * self.h_safe * (sox + sfx)
        s_mom_y = -g * self.h_safe * (soy + sfy)
        return jnp.stack([s_mass, s_mom_x, s_mom_y], axis=-1)