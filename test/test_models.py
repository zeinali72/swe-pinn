"""Unit tests for src.models (FourierPINN, MLP, DGMNetwork, init_model)."""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import unittest

import jax
import jax.numpy as jnp

import src.config as config_module
config_module.DTYPE = jnp.float32

from src.models.pinn import FourierPINN, MLP, DGMNetwork
from src.models.factory import init_model


def _make_config(model_name='MLP', **model_overrides):
    """Build a minimal config dict for model initialization."""
    model_cfg = {
        'name': model_name,
        'width': 16,
        'depth': 2,
        'output_dim': 3,
        'bias_init': 0.0,
    }
    if model_name == 'FourierPINN':
        model_cfg.update({
            'ff_dims': 32,
            'fourier_scale': 1.0,
        })
    model_cfg.update(model_overrides)
    return {
        'model': model_cfg,
        'domain': {
            'lx': 100.0,
            'ly': 10.0,
            't_final': 60.0,
        },
    }


class TestMLPForwardPass(unittest.TestCase):
    """Tests for MLP forward pass."""

    def test_output_shape(self):
        """MLP: input (N, 3) produces output (N, 3)."""
        config = _make_config('MLP')
        model = MLP(config=config)
        key = jax.random.PRNGKey(0)
        variables = model.init(key, jnp.zeros((1, 3)))
        x = jax.random.uniform(key, (8, 3))
        y = model.apply(variables, x, train=False)
        self.assertEqual(y.shape, (8, 3))

    def test_output_finite(self):
        """MLP output values are finite (no NaN/Inf)."""
        config = _make_config('MLP')
        model = MLP(config=config)
        key = jax.random.PRNGKey(1)
        variables = model.init(key, jnp.zeros((1, 3)))
        x = jax.random.uniform(key, (8, 3))
        y = model.apply(variables, x, train=False)
        self.assertTrue(jnp.all(jnp.isfinite(y)), f"MLP output contains NaN/Inf: {y}")


class TestFourierPINNForwardPass(unittest.TestCase):
    """Tests for FourierPINN forward pass."""

    def test_output_shape(self):
        """FourierPINN: input (N, 3) produces output (N, 3)."""
        config = _make_config('FourierPINN')
        model = FourierPINN(config=config)
        key = jax.random.PRNGKey(0)
        variables = model.init(key, jnp.zeros((1, 3)))
        x = jax.random.uniform(key, (8, 3))
        y = model.apply(variables, x, train=False)
        self.assertEqual(y.shape, (8, 3))

    def test_output_finite(self):
        """FourierPINN output values are finite."""
        config = _make_config('FourierPINN')
        model = FourierPINN(config=config)
        key = jax.random.PRNGKey(1)
        variables = model.init(key, jnp.zeros((1, 3)))
        x = jax.random.uniform(key, (8, 3))
        y = model.apply(variables, x, train=False)
        self.assertTrue(jnp.all(jnp.isfinite(y)),
                        f"FourierPINN output contains NaN/Inf: {y}")


class TestDGMNetworkForwardPass(unittest.TestCase):
    """Tests for DGMNetwork forward pass."""

    def test_output_shape(self):
        """DGMNetwork: input (N, 3) produces output (N, 3)."""
        config = _make_config('DGMNetwork')
        model = DGMNetwork(config=config)
        key = jax.random.PRNGKey(0)
        variables = model.init(key, jnp.zeros((1, 3)))
        x = jax.random.uniform(key, (8, 3))
        y = model.apply(variables, x, train=False)
        self.assertEqual(y.shape, (8, 3))

    def test_output_finite(self):
        """DGMNetwork output values are finite."""
        config = _make_config('DGMNetwork')
        model = DGMNetwork(config=config)
        key = jax.random.PRNGKey(1)
        variables = model.init(key, jnp.zeros((1, 3)))
        x = jax.random.uniform(key, (8, 3))
        y = model.apply(variables, x, train=False)
        self.assertTrue(jnp.all(jnp.isfinite(y)),
                        f"DGMNetwork output contains NaN/Inf: {y}")


class TestInitModelFactory(unittest.TestCase):
    """Tests for the init_model factory function."""

    def test_returns_fourier_pinn(self):
        """init_model with FourierPINN returns a FourierPINN instance."""
        config = _make_config('FourierPINN')
        key = jax.random.PRNGKey(0)
        model, params = init_model(FourierPINN, key, config)
        self.assertIsInstance(model, FourierPINN)
        self.assertIn('params', params)

    def test_returns_mlp(self):
        """init_model with MLP returns an MLP instance."""
        config = _make_config('MLP')
        key = jax.random.PRNGKey(0)
        model, params = init_model(MLP, key, config)
        self.assertIsInstance(model, MLP)
        self.assertIn('params', params)

    def test_returns_dgm(self):
        """init_model with DGMNetwork returns a DGMNetwork instance."""
        config = _make_config('DGMNetwork')
        key = jax.random.PRNGKey(0)
        model, params = init_model(DGMNetwork, key, config)
        self.assertIsInstance(model, DGMNetwork)
        self.assertIn('params', params)

    def test_factory_params_produce_valid_output(self):
        """Params from init_model produce finite forward-pass output."""
        config = _make_config('MLP')
        key = jax.random.PRNGKey(0)
        model, params = init_model(MLP, key, config)
        x = jax.random.uniform(key, (4, 3))
        y = model.apply({'params': params['params']}, x, train=False)
        self.assertEqual(y.shape, (4, 3))
        self.assertTrue(jnp.all(jnp.isfinite(y)))


if __name__ == '__main__':
    unittest.main()
