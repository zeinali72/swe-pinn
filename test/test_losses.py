"""Unit tests for src.losses (PDE, boundary, data, composite)."""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import sys
import types
import unittest

import jax
import jax.numpy as jnp
from flax import linen as nn

import src.config as config_module
config_module.DTYPE = jnp.float32

# --- Stub out src.data to avoid the missing src.data.paths import ---
# The PDE loss module imports ``from src.data import bathymetry_fn``.
# ``src.data.__init__`` tries to import ``src.data.paths`` which does not
# exist on this branch.  We inject a lightweight stub so the import chain
# succeeds without touching the filesystem.
_data_stub = types.ModuleType("src.data")
_data_stub.bathymetry_fn = lambda x, y: (
    jnp.zeros_like(x),  # bed elevation
    jnp.zeros_like(x),  # bed_grad_x
    jnp.zeros_like(x),  # bed_grad_y
)
# Only inject if not already importable
if "src.data" not in sys.modules:
    sys.modules["src.data"] = _data_stub
    # Also add sub-modules referenced by __init__ so nothing breaks
    for _sub in ("src.data.sampling", "src.data.batching", "src.data.loading",
                 "src.data.bathymetry", "src.data.irregular", "src.data.paths"):
        if _sub not in sys.modules:
            sys.modules[_sub] = types.ModuleType(_sub)

from src.losses.pde import compute_neg_h_loss, compute_ic_loss, compute_pde_loss
from src.losses.boundary import (
    loss_boundary_dirichlet_h,
    loss_boundary_dirichlet_hu,
    loss_boundary_dirichlet_hv,
)
from src.losses.composite import total_loss


class _DummyModel(nn.Module):
    """A trivial model that returns zeros, for testing loss functions."""
    @nn.compact
    def __call__(self, x, train=True):
        return jnp.zeros((x.shape[0], 3))


class _ConstantModel(nn.Module):
    """A model that returns a fixed constant output."""
    value: float = 1.0

    @nn.compact
    def __call__(self, x, train=True):
        return jnp.full((x.shape[0], 3), self.value)


class _NegativeHModel(nn.Module):
    """A model where h (first output) is negative."""
    @nn.compact
    def __call__(self, x, train=True):
        batch_size = x.shape[0]
        h = jnp.full((batch_size,), -0.5)
        hu = jnp.zeros((batch_size,))
        hv = jnp.zeros((batch_size,))
        return jnp.stack([h, hu, hv], axis=-1)


def _init_dummy(model):
    """Initialize a model with a dummy key and return params dict."""
    key = jax.random.PRNGKey(0)
    variables = model.init(key, jnp.zeros((1, 3)))
    return {'params': variables.get('params', {})}


class TestNegHLoss(unittest.TestCase):
    """Tests for compute_neg_h_loss."""

    def test_positive_h_gives_zero(self):
        """Loss is 0 when all water depths are positive."""
        model = _ConstantModel(value=1.0)
        params = _init_dummy(model)
        points = jnp.ones((5, 3))
        loss = compute_neg_h_loss(model, params, points)
        self.assertAlmostEqual(float(loss), 0.0, places=8)

    def test_negative_h_gives_positive_loss(self):
        """Loss is positive when water depth is negative."""
        model = _NegativeHModel()
        params = _init_dummy(model)
        points = jnp.ones((5, 3))
        loss = compute_neg_h_loss(model, params, points)
        self.assertGreater(float(loss), 0.0)


class TestICLoss(unittest.TestCase):
    """Tests for compute_ic_loss."""

    def test_zero_prediction_gives_zero_loss(self):
        """IC loss is 0 when model predicts zeros at t=0."""
        model = _DummyModel()
        params = _init_dummy(model)
        ic_batch = jnp.zeros((10, 3))
        loss = compute_ic_loss(model, params, ic_batch)
        self.assertAlmostEqual(float(loss), 0.0, places=8)

    def test_nonzero_prediction_gives_positive_loss(self):
        """IC loss is positive when model does not predict zeros."""
        model = _ConstantModel(value=0.5)
        params = _init_dummy(model)
        ic_batch = jnp.zeros((10, 3))
        loss = compute_ic_loss(model, params, ic_batch)
        self.assertGreater(float(loss), 0.0)


class TestDirichletBCLoss(unittest.TestCase):
    """Tests for Dirichlet boundary loss functions."""

    def test_dirichlet_h_zero_when_matching(self):
        """loss_boundary_dirichlet_h returns 0 when prediction matches target."""
        model = _ConstantModel(value=2.0)
        params = _init_dummy(model)
        batch = jnp.ones((5, 3))
        h_target = jnp.full((5,), 2.0)
        loss = loss_boundary_dirichlet_h(model, params, batch, h_target)
        self.assertAlmostEqual(float(loss), 0.0, places=8)

    def test_dirichlet_h_positive_when_mismatched(self):
        """loss_boundary_dirichlet_h returns positive when target differs."""
        model = _ConstantModel(value=2.0)
        params = _init_dummy(model)
        batch = jnp.ones((5, 3))
        h_target = jnp.full((5,), 0.0)
        loss = loss_boundary_dirichlet_h(model, params, batch, h_target)
        self.assertGreater(float(loss), 0.0)

    def test_dirichlet_hu_zero_when_matching(self):
        """loss_boundary_dirichlet_hu returns 0 when prediction matches target."""
        model = _ConstantModel(value=3.0)
        params = _init_dummy(model)
        batch = jnp.ones((5, 3))
        hu_target = jnp.full((5,), 3.0)
        loss = loss_boundary_dirichlet_hu(model, params, batch, hu_target)
        self.assertAlmostEqual(float(loss), 0.0, places=8)

    def test_dirichlet_hv_zero_when_matching(self):
        """loss_boundary_dirichlet_hv returns 0 when prediction matches target."""
        model = _ConstantModel(value=4.0)
        params = _init_dummy(model)
        batch = jnp.ones((5, 3))
        hv_target = jnp.full((5,), 4.0)
        loss = loss_boundary_dirichlet_hv(model, params, batch, hv_target)
        self.assertAlmostEqual(float(loss), 0.0, places=8)


class TestTotalLoss(unittest.TestCase):
    """Tests for the total_loss weighted sum."""

    def test_weighted_sum_correct(self):
        """total_loss correctly computes the weighted sum."""
        terms = {
            'pde': jnp.array(1.0),
            'ic': jnp.array(2.0),
            'bc': jnp.array(3.0),
        }
        weights = {
            'pde': 10.0,
            'ic': 5.0,
            'bc': 1.0,
        }
        result = total_loss(terms, weights)
        expected = 10.0 * 1.0 + 5.0 * 2.0 + 1.0 * 3.0  # 23.0
        self.assertAlmostEqual(float(result), expected, places=6)

    def test_missing_weight_ignored(self):
        """Terms without a matching weight are ignored."""
        terms = {
            'pde': jnp.array(1.0),
            'extra': jnp.array(100.0),
        }
        weights = {
            'pde': 2.0,
        }
        result = total_loss(terms, weights)
        self.assertAlmostEqual(float(result), 2.0, places=6)

    def test_empty_terms(self):
        """Empty terms dict gives zero loss."""
        result = total_loss({}, {'pde': 1.0})
        self.assertAlmostEqual(float(result), 0.0, places=6)


class TestPDELossRunnable(unittest.TestCase):
    """Smoke test: compute_pde_loss runs without error and returns a finite scalar."""

    def test_pde_loss_finite(self):
        """PDE loss should return a finite scalar for a simple model."""
        from src.models.pinn import MLP

        config = {
            'model': {
                'name': 'MLP',
                'width': 8,
                'depth': 1,
                'output_dim': 3,
                'bias_init': 0.0,
            },
            'domain': {
                'lx': 100.0,
                'ly': 10.0,
                't_final': 60.0,
            },
            'physics': {
                'u_const': 0.0,
                'n_manning': 0.0,
                'inflow': None,
                'g': 9.81,
            },
            'numerics': {
                'eps': 1e-6,
            },
        }

        model = MLP(config=config)
        key = jax.random.PRNGKey(42)
        variables = model.init(key, jnp.zeros((1, 3)))
        params = {'params': variables['params']}

        pde_batch = jax.random.uniform(key, (10, 3))
        loss = compute_pde_loss(model, params, pde_batch, config)
        self.assertTrue(jnp.isfinite(loss), f"PDE loss is not finite: {loss}")


if __name__ == '__main__':
    unittest.main()
