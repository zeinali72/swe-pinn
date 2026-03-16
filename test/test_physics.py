"""Unit tests for src.physics.swe.SWEPhysics."""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import unittest

import jax.numpy as jnp

from src.physics.swe import SWEPhysics


class TestFluxJacobians(unittest.TestCase):
    """Tests for SWEPhysics.flux_jac()."""

    def test_quiescent_water(self):
        """For quiescent water (h=1, hu=0, hv=0), verify analytical Jacobians."""
        g = 9.81
        eps = 1e-6
        U = jnp.array([[1.0, 0.0, 0.0]])  # h=1, hu=0, hv=0
        physics = SWEPhysics(U, eps=eps)
        F, G = physics.flux_jac(g)

        # Expected F Jacobian for quiescent water:
        # Row 0: [0, 1, 0]
        # Row 1: [-u^2 + g*h, 2*u, 0] = [g*1, 0, 0] = [9.81, 0, 0]
        # Row 2: [-u*v, v, u] = [0, 0, 0]
        F_expected = jnp.array([[[0.0, 1.0, 0.0],
                                  [g * 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0]]])

        # Expected G Jacobian for quiescent water:
        # Row 0: [0, 0, 1]
        # Row 1: [-u*v, v, u] = [0, 0, 0]
        # Row 2: [-v^2 + g*h, 0, 2*v] = [g*1, 0, 0] = [9.81, 0, 0]
        G_expected = jnp.array([[[0.0, 0.0, 1.0],
                                  [0.0, 0.0, 0.0],
                                  [g * 1.0, 0.0, 0.0]]])

        self.assertTrue(jnp.allclose(F, F_expected, atol=1e-6),
                        f"F Jacobian mismatch:\n{F}\nvs expected:\n{F_expected}")
        self.assertTrue(jnp.allclose(G, G_expected, atol=1e-6),
                        f"G Jacobian mismatch:\n{G}\nvs expected:\n{G_expected}")

    def test_flux_jac_shape(self):
        """Jacobians should have shape (N, 3, 3) for N points."""
        U = jnp.array([[2.0, 0.5, 0.3],
                        [1.0, 0.1, 0.2]])
        physics = SWEPhysics(U, eps=1e-6)
        F, G = physics.flux_jac(g=9.81)
        self.assertEqual(F.shape, (2, 3, 3))
        self.assertEqual(G.shape, (2, 3, 3))


class TestSource(unittest.TestCase):
    """Tests for SWEPhysics.source()."""

    def test_flat_no_friction_no_inflow(self):
        """Source returns zeros for flat domain, no friction, no inflow."""
        U = jnp.array([[1.0, 0.0, 0.0]])
        physics = SWEPhysics(U, eps=1e-6)
        S = physics.source(g=9.81, n_manning=0.0, inflow=None)

        # s_mass = 0 (no inflow), s_mom_x = 0 (no slope, no friction), s_mom_y = 0
        expected = jnp.array([[0.0, 0.0, 0.0]])
        self.assertTrue(jnp.allclose(S, expected, atol=1e-8),
                        f"Source mismatch:\n{S}\nvs expected:\n{expected}")

    def test_manning_friction(self):
        """Manning friction produces non-zero momentum source terms."""
        U = jnp.array([[1.0, 1.0, 0.5]])  # h=1, hu=1, hv=0.5
        physics = SWEPhysics(U, eps=1e-6)
        S = physics.source(g=9.81, n_manning=0.03, inflow=None)

        # Mass source should be zero (no inflow)
        self.assertAlmostEqual(float(S[0, 0]), 0.0, places=6)
        # Momentum sources should be non-zero due to friction
        self.assertNotAlmostEqual(float(S[0, 1]), 0.0, places=6,
                                  msg="s_mom_x should be non-zero with friction")
        self.assertNotAlmostEqual(float(S[0, 2]), 0.0, places=6,
                                  msg="s_mom_y should be non-zero with friction")

    def test_bed_slope(self):
        """Bed slope terms appear in source when bed gradients are non-zero."""
        U = jnp.array([[1.0, 0.0, 0.0]])  # quiescent
        physics = SWEPhysics(U, eps=1e-6)
        bed_grad_x = jnp.array([0.01])
        bed_grad_y = jnp.array([0.02])
        S = physics.source(g=9.81, n_manning=0.0, inflow=None,
                           bed_grad_x=bed_grad_x, bed_grad_y=bed_grad_y)

        # Mass source should be zero
        self.assertAlmostEqual(float(S[0, 0]), 0.0, places=6)
        # Momentum sources should be non-zero due to bed slope
        # sox = -bed_grad_x = -0.01, s_mom_x = -g * h * (sox + sfx)
        # = -9.81 * 1.0 * (-0.01 + 0) = 0.0981
        self.assertNotAlmostEqual(float(S[0, 1]), 0.0, places=6,
                                  msg="s_mom_x should be non-zero with bed slope")
        self.assertNotAlmostEqual(float(S[0, 2]), 0.0, places=6,
                                  msg="s_mom_y should be non-zero with bed slope")

    def test_inflow_source(self):
        """Non-None inflow produces non-zero mass source."""
        U = jnp.array([[1.0, 0.0, 0.0]])
        physics = SWEPhysics(U, eps=1e-6)
        S = physics.source(g=9.81, n_manning=0.0, inflow=0.001)

        # Mass source should equal inflow rate
        self.assertAlmostEqual(float(S[0, 0]), 0.001, places=6)


if __name__ == '__main__':
    unittest.main()
