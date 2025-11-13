# src/physics_deeponet.py
# This file is parallel to physics.py, but adapted for DeepONet.
# It accepts parameters 'n_manning' and 'u_const' as (N,) arrays.

import jax.numpy as jnp
from typing import Tuple

# h_exact from the original physics.py is already compatible with array inputs
from src.physics import h_exact 

class SWEPhysics_DeepONet:
    """
    Compute terms for the 2D SWE, adapted for DeepONet.
    Accepts physics parameters (n_manning, u_const) as (N,) arrays.
    """
    def __init__(self, U: jnp.ndarray, eps: float, bed_elevation: jnp.ndarray = None):
        h = U[..., 0]
        self.hu = U[..., 1]
        self.hv = U[..., 2]
        
        # h_safe must be (N,)
        self.h_safe = jnp.maximum(h, eps)
        # u and v will be (N,)
        self.u = self.hu / self.h_safe
        self.v = self.hv / self.h_safe
        self.bed = bed_elevation

    def flux_jac(self, g: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute analytical Jacobians of SWE fluxes. (Identical to PINN version)"""
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

    def source(self, g: float, n_manning: jnp.ndarray, inflow: float) -> jnp.ndarray:
        """
        Compute source terms for SWE.
        Accepts n_manning as an array of shape (N,).
        """
        vel = jnp.sqrt(self.u**2 + self.v**2)
        
        # n_manning, u, v, vel, h_safe are all (N,)
        # sfx and sfy will be (N,)
        sfx = n_manning**2 * self.u * vel / (self.h_safe**(4 / 3))
        sfy = n_manning**2 * self.v * vel / (self.h_safe**(4 / 3))
        sox = soy = 0.0
        if self.bed is not None:
            # This part is complex if bed elevation depends on params.
            # For the analytical case, bed is flat, so gradients are 0.
            sox, soy = 0.0, 0.0
            # sox, soy = -jnp.gradient(self.bed, axis=-2), -jnp.gradient(self.bed, axis=-1)

        R = 0.0 if inflow is None else inflow
        
        # s_mass, s_mom_x, s_mom_y are (N,)
        s_mass = R * jnp.ones_like(self.h_safe)
        s_mom_x = -g * self.h_safe * (sox + sfx)
        s_mom_y = -g * self.h_safe * (soy + sfy)
        
        # Return stacked source terms, shape (N, 3)
        return jnp.stack([s_mass, s_mom_x, s_mom_y], axis=-1)