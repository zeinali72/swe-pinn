"""Non-dimensionalization of the 2D Shallow Water Equations.

Transforms the conservative-form SWE into a dimensionless system where all
residual terms are O(1), eliminating gravity from the flux terms entirely.

Characteristic scales
---------------------
L0 : Length scale — max(Lx, Ly) from domain geometry.
H0 : Depth scale — reference depth (configurable, default 1.0 m).
U0 : Velocity scale — shallow-water celerity sqrt(g * H0).
T0 : Time scale — advective timescale L0 / U0.

The single dimensionless group is the friction number:
    C_f = g * n_manning^2 * L0 / H0^(4/3)
"""

import jax.numpy as jnp
from flax.core import FrozenDict


class SWEScaler:
    """Handles non-dimensionalization of inputs, outputs, and physics parameters.

    Parameters
    ----------
    cfg : FrozenDict or dict
        Config with keys ``domain.lx``, ``domain.ly``, ``domain.t_final``,
        ``physics.g``, ``physics.n_manning``, and optionally ``scaling.H0``.
    """

    def __init__(self, cfg):
        domain = cfg["domain"]
        physics = cfg["physics"]
        scaling_cfg = cfg.get("scaling", {})

        g = float(physics["g"])
        n_manning = float(physics["n_manning"])

        self.L0 = float(max(domain["lx"], domain["ly"]))
        self.H0 = float(scaling_cfg.get("H0", 1.0))
        self.U0 = float(jnp.sqrt(g * self.H0))
        self.T0 = self.L0 / self.U0
        self.Cf = g * n_manning ** 2 * self.L0 / self.H0 ** (4.0 / 3.0)

        # Pre-compute output scale product for hu/hv
        self._HU0 = self.H0 * self.U0

    # ------------------------------------------------------------------
    # Input scaling
    # ------------------------------------------------------------------

    def scale_inputs(self, pts: jnp.ndarray) -> jnp.ndarray:
        """Scale (x, y, t) coordinates to dimensionless form.

        Parameters
        ----------
        pts : jnp.ndarray, shape (..., 3)
            Dimensional coordinates [x, y, t].

        Returns
        -------
        jnp.ndarray, shape (..., 3)
            Dimensionless coordinates [x*, y*, t*].
        """
        scales = jnp.array([self.L0, self.L0, self.T0], dtype=pts.dtype)
        return pts / scales

    def scale_range(self, lo: float, hi: float, dim: str) -> tuple[float, float]:
        """Scale a dimensional range to dimensionless form.

        Parameters
        ----------
        lo, hi : float
            Dimensional range bounds.
        dim : str
            One of ``'x'``, ``'y'``, ``'t'``.
        """
        s = self.T0 if dim == "t" else self.L0
        return lo / s, hi / s

    # ------------------------------------------------------------------
    # Output scaling
    # ------------------------------------------------------------------

    def scale_outputs(self, h: jnp.ndarray, hu: jnp.ndarray, hv: jnp.ndarray):
        """Scale dimensional [h, hu, hv] to dimensionless form."""
        return h / self.H0, hu / self._HU0, hv / self._HU0

    def scale_output_array(self, U: jnp.ndarray) -> jnp.ndarray:
        """Scale a stacked output array [..., 3] (h, hu, hv) to dimensionless form."""
        scales = jnp.array([self.H0, self._HU0, self._HU0], dtype=U.dtype)
        return U / scales

    def unscale_outputs(self, h_star: jnp.ndarray, hu_star: jnp.ndarray, hv_star: jnp.ndarray):
        """Convert dimensionless predictions back to dimensional form."""
        return h_star * self.H0, hu_star * self._HU0, hv_star * self._HU0

    def unscale_output_array(self, U_star: jnp.ndarray) -> jnp.ndarray:
        """Unscale a stacked output array [..., 3] back to dimensional form."""
        scales = jnp.array([self.H0, self._HU0, self._HU0], dtype=U_star.dtype)
        return U_star * scales

    # ------------------------------------------------------------------
    # Bed / bathymetry scaling
    # ------------------------------------------------------------------

    def scale_bed(self, z_b: jnp.ndarray) -> jnp.ndarray:
        """Scale bed elevation to dimensionless form."""
        return z_b / self.H0

    def scale_bed_gradient(self, dz_dx: jnp.ndarray, dz_dy: jnp.ndarray):
        """Scale bed gradients: dz*/dx* = (L0/H0) * dz/dx."""
        ratio = self.L0 / self.H0
        return dz_dx * ratio, dz_dy * ratio

    # ------------------------------------------------------------------
    # Physics parameter
    # ------------------------------------------------------------------

    @property
    def dimensionless_friction(self) -> float:
        """Return the dimensionless friction number C_f."""
        return self.Cf

    # ------------------------------------------------------------------
    # Config for the non-dimensional PDE
    # ------------------------------------------------------------------

    def nondim_physics_config(self, base_config) -> FrozenDict:
        """Build a config for the non-dimensional PDE.

        The returned config sets ``g = 1.0`` (absorbed into scaling),
        ``n_manning = 0.0`` (friction absorbed into ``Cf``), and adds
        ``Cf`` so the SWE source term uses the dimensionless friction
        number exclusively.

        The original dimensional ``n_manning`` and ``u_const`` are
        preserved under ``physics.dimensional`` so that experiment
        scripts can still compute analytical BC targets in dimensional
        space before scaling.
        """
        cfg_dict = dict(base_config)
        physics = dict(cfg_dict.get("physics", {}))
        # Preserve original values for analytical BC computations
        physics["dimensional"] = {
            "n_manning": physics["n_manning"],
            "u_const": physics.get("u_const"),
            "g": physics["g"],
        }
        physics["g"] = 1.0
        physics["n_manning"] = 0.0
        physics["Cf"] = self.Cf
        cfg_dict["physics"] = physics
        return FrozenDict(cfg_dict)

    def summary(self) -> str:
        """Return a human-readable summary of the scaling parameters."""
        return (
            f"SWE Non-dimensionalization:\n"
            f"  L0 = {self.L0:.2f} m\n"
            f"  H0 = {self.H0:.4f} m\n"
            f"  U0 = {self.U0:.4f} m/s (celerity)\n"
            f"  T0 = {self.T0:.4f} s\n"
            f"  Cf = {self.Cf:.6f} (dimensionless friction)"
        )
