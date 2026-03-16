"""Unit tests for src.physics.swe.SWEPhysics."""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import unittest

import jax.numpy as jnp

from src.physics.swe import SWEPhysics
from src.physics.analytical import h_exact, hu_exact, hv_exact


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


class TestAnalyticalSolutions(unittest.TestCase):
    """Tests for analytical dam-break solutions (h, hu, hv)."""

    def setUp(self):
        self.n_manning = 0.03
        self.u_const = 0.29

    def test_h_exact_behind_wave_front(self):
        """h > 0 behind the wave front (u*t > x)."""
        x = jnp.array([0.0, 10.0])
        t = jnp.array([100.0, 100.0])
        h = h_exact(x, t, self.n_manning, self.u_const)
        self.assertTrue(jnp.all(h > 0), f"h should be positive behind wave front: {h}")

    def test_h_exact_ahead_of_wave_front(self):
        """h = 0 ahead of the wave front (u*t < x)."""
        x = jnp.array([1000.0])
        t = jnp.array([1.0])  # wave front at u*t = 0.29
        h = h_exact(x, t, self.n_manning, self.u_const)
        self.assertAlmostEqual(float(h[0]), 0.0, places=10)

    def test_hu_exact_equals_h_times_u(self):
        """hu = h * u_const everywhere."""
        x = jnp.array([0.0, 5.0, 50.0, 500.0])
        t = jnp.array([100.0, 100.0, 100.0, 100.0])
        h = h_exact(x, t, self.n_manning, self.u_const)
        hu = hu_exact(x, t, self.n_manning, self.u_const)
        expected = h * self.u_const
        self.assertTrue(jnp.allclose(hu, expected, atol=1e-10),
                        f"hu should equal h * u_const: {hu} vs {expected}")

    def test_hu_exact_zero_where_h_zero(self):
        """hu = 0 where h = 0 (ahead of wave front)."""
        x = jnp.array([1000.0])
        t = jnp.array([1.0])
        hu = hu_exact(x, t, self.n_manning, self.u_const)
        self.assertAlmostEqual(float(hu[0]), 0.0, places=10)

    def test_hv_exact_always_zero(self):
        """hv = 0 everywhere (1D problem, no y-velocity).

        This is a regression guard: hv_exact returns zeros_like(x) by
        definition for the 1D dam-break. The test ensures it isn't
        accidentally changed to return non-zero values.
        """
        x = jnp.array([0.0, 10.0, 100.0])
        t = jnp.array([100.0, 100.0, 100.0])
        hv = hv_exact(x, t, self.n_manning, self.u_const)
        self.assertTrue(jnp.allclose(hv, 0.0),
                        f"hv should be zero everywhere: {hv}")

    def test_initial_condition_t_zero(self):
        """At t=0, h=0 and hu=0 for all x > 0 (dry domain before dam-break)."""
        x = jnp.array([1.0, 10.0, 100.0, 500.0])
        t = jnp.zeros_like(x)
        h = h_exact(x, t, self.n_manning, self.u_const)
        hu = hu_exact(x, t, self.n_manning, self.u_const)
        hv = hv_exact(x, t, self.n_manning, self.u_const)
        self.assertTrue(jnp.allclose(h, 0.0, atol=1e-12),
                        f"h should be 0 at t=0 for x>0: {h}")
        self.assertTrue(jnp.allclose(hu, 0.0, atol=1e-12),
                        f"hu should be 0 at t=0 for x>0: {hu}")
        self.assertTrue(jnp.allclose(hv, 0.0, atol=1e-12),
                        f"hv should be 0 at t=0: {hv}")

    def test_h_exact_at_origin(self):
        """h at x=0 grows with time (dam-break spreading)."""
        x = jnp.array([0.0, 0.0])
        t = jnp.array([100.0, 200.0])
        h = h_exact(x, t, self.n_manning, self.u_const)
        self.assertGreater(float(h[1]), float(h[0]),
                           "h at origin should increase with time")


if __name__ == '__main__':
    unittest.main()
