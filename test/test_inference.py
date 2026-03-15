"""Unit and integration tests for the inference pipeline."""
import os
import pickle
import shutil
import sys
import tempfile
import unittest

import yaml

os.environ["JAX_PLATFORM_NAME"] = "cpu"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Metric unit tests
# ---------------------------------------------------------------------------


class TestPeakMetrics(unittest.TestCase):
    def test_r_squared_perfect(self):
        from src.metrics.peak import r_squared
        x = jnp.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(r_squared(x, x), 1.0, places=5)

    def test_r_squared_mean_prediction(self):
        from src.metrics.peak import r_squared
        true = jnp.array([1.0, 2.0, 3.0, 4.0])
        pred = jnp.full_like(true, jnp.mean(true))
        self.assertAlmostEqual(r_squared(pred, true), 0.0, places=5)

    def test_peak_depth_error(self):
        from src.metrics.peak import peak_depth_error
        pred = jnp.array([0.0, 0.5, 0.9, 0.2])
        true = jnp.array([0.0, 0.5, 1.0, 0.2])
        self.assertAlmostEqual(peak_depth_error(pred, true), 0.1, places=5)

    def test_time_to_peak_error(self):
        from src.metrics.peak import time_to_peak_error
        pred = jnp.array([0.0, 0.5, 1.0, 0.2])
        true = jnp.array([0.0, 1.0, 0.5, 0.2])
        t = jnp.array([0.0, 1.0, 2.0, 3.0])
        self.assertAlmostEqual(time_to_peak_error(pred, true, t), 1.0, places=5)

    def test_rmse_mae_ratio_uniform(self):
        from src.metrics.peak import rmse_mae_ratio
        pred = jnp.array([1.0, 2.0, 3.0])
        true = jnp.array([1.1, 2.1, 3.1])
        ratio = rmse_mae_ratio(pred, true)
        # Uniform error -> ratio should be ~1.0
        self.assertAlmostEqual(ratio, 1.0, places=5)


class TestNegativeDepth(unittest.TestCase):
    def test_no_negatives(self):
        from src.metrics.negative_depth import negative_depth_stats
        h = jnp.array([0.0, 0.1, 0.5, 1.0])
        stats = negative_depth_stats(h)
        self.assertEqual(stats["count"], 0)
        self.assertAlmostEqual(stats["fraction"], 0.0)

    def test_with_negatives(self):
        from src.metrics.negative_depth import negative_depth_stats
        h = jnp.array([-0.1, 0.1, -0.5, 1.0])
        stats = negative_depth_stats(h)
        self.assertEqual(stats["count"], 2)
        self.assertAlmostEqual(stats["fraction"], 0.5)
        self.assertAlmostEqual(stats["min_h"], -0.5, places=5)


class TestFloodExtent(unittest.TestCase):
    def test_perfect_match(self):
        from src.metrics.flood_extent import flood_extent_metrics
        h = jnp.array([0.0, 0.02, 0.1, 0.5])
        result = flood_extent_metrics(h, h, thresholds=(0.01,))
        self.assertAlmostEqual(result["threshold_0.01"]["csi"], 1.0)
        self.assertAlmostEqual(result["threshold_0.01"]["hit_rate"], 1.0)
        self.assertAlmostEqual(result["threshold_0.01"]["far"], 0.0)

    def test_complete_miss(self):
        from src.metrics.flood_extent import flood_extent_metrics
        pred = jnp.array([0.0, 0.0, 0.0, 0.0])
        true = jnp.array([0.0, 0.02, 0.1, 0.5])
        result = flood_extent_metrics(pred, true, thresholds=(0.01,))
        self.assertAlmostEqual(result["threshold_0.01"]["hit_rate"], 0.0)


class TestConservation(unittest.TestCase):
    def test_volume_balance_returns_dict(self):
        from src.metrics.conservation import volume_balance
        h = jnp.array([0.1, 0.2, 0.15, 0.1, 0.2, 0.15])
        coords = jnp.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1],
        ], dtype=jnp.float32)
        bounds = {"lx": 1.0, "ly": 1.0, "t_final": 1.0}
        result = volume_balance(h, coords, bounds, n_time_steps=2)
        self.assertIn("max_mass_error_pct", result)
        self.assertIn("volume_time_series", result)


class TestBoundary(unittest.TestCase):
    def test_ic_accuracy_zero_prediction(self):
        from src.metrics.boundary import initial_condition_accuracy
        # Create a trivial model that returns zeros
        import flax.linen as nn

        class TinyModel(nn.Module):
            @nn.compact
            def __call__(self, x, train=False):
                return jnp.zeros((x.shape[0], 3))

        model = TinyModel()
        variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, 3)))
        # TinyModel has no trainable params, so variables may not contain 'params'
        params = {"params": variables.get("params", {})}
        ic_coords = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32)
        result = initial_condition_accuracy(model, params, ic_coords)
        self.assertAlmostEqual(result["rmse"], 0.0, places=5)

    def test_ic_accuracy_empty(self):
        from src.metrics.boundary import initial_condition_accuracy
        result = initial_condition_accuracy(None, None, jnp.zeros((0, 3)))
        self.assertAlmostEqual(result["rmse"], 0.0)


class TestDecomposition(unittest.TestCase):
    def test_temporal_decomposition_returns_phases(self):
        from src.metrics.decomposition import temporal_decomposition
        n = 100
        t = jnp.linspace(0, 10, n)
        # Triangular pulse
        h_true = jnp.where(t < 5, t / 5, (10 - t) / 5)
        h_pred = h_true + 0.01
        result = temporal_decomposition(h_pred, h_true, t)
        self.assertIn("rising", result)
        self.assertIn("peak", result)
        self.assertIn("recession", result)

    def test_spatial_decomposition_returns_regions(self):
        from src.metrics.decomposition import spatial_decomposition
        n = 200
        key = jax.random.PRNGKey(42)
        coords = jax.random.uniform(key, (n, 3))
        coords = coords.at[:, 0].multiply(10.0)
        coords = coords.at[:, 1].multiply(5.0)
        true = jax.random.uniform(jax.random.PRNGKey(1), (n, 3))
        pred = true + 0.01
        bounds = {"lx": 10.0, "ly": 5.0, "x_min": 0.0, "y_min": 0.0}
        result = spatial_decomposition(pred, true, coords, bounds)
        self.assertIn("shock", result)
        self.assertIn("boundary", result)
        self.assertIn("interior", result)


# ---------------------------------------------------------------------------
# Reporting tests
# ---------------------------------------------------------------------------


class TestReporting(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_save_yaml_summary(self):
        from src.inference.reporting import save_yaml_summary
        results = {"accuracy": {"nse_h": 0.95}, "n_points": 1000}
        path = save_yaml_summary(results, self.tmpdir)
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            loaded = yaml.safe_load(f)
        self.assertEqual(loaded["n_points"], 1000)

    def test_save_raw_predictions(self):
        from src.inference.reporting import save_raw_predictions
        from src.inference.context import InferenceContext

        ctx = InferenceContext(
            config={}, model=None, params={}, predictor=None,
            val_coords=jnp.zeros((5, 3)),
            val_targets=jnp.zeros((5, 3)),
            predictions=jnp.zeros((5, 3)),
            experiment_name="test",
            domain_bounds={"lx": 1, "ly": 1, "t_final": 1},
            experiment_meta={},
        )
        path = save_raw_predictions(ctx, self.tmpdir)
        self.assertTrue(os.path.exists(path))
        data = np.load(path)
        self.assertEqual(data["predictions"].shape, (5, 3))

    def test_print_text_report(self):
        from src.inference.reporting import print_text_report
        results = {
            "checkpoint": "test",
            "n_points": 100,
            "inference_time_seconds": 0.5,
            "accuracy": {"nse_h": 0.95, "rmse_h": 0.01, "negative_depth": {"fraction": 0.0, "min_h": 0.0}},
        }
        # Should not raise
        print_text_report(results, self.tmpdir)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "report.txt")))

    def test_save_comparison_table(self):
        from src.inference.reporting import save_comparison_table
        all_results = {
            "best_nse": {
                "accuracy": {"nse_h": 0.95, "rmse_h": 0.01, "negative_depth": {"fraction": 0.0}},
                "inference_time_seconds": 0.5,
            },
            "best_loss": {
                "accuracy": {"nse_h": 0.90, "rmse_h": 0.02, "negative_depth": {"fraction": 0.01}},
                "inference_time_seconds": 0.6,
            },
        }
        save_comparison_table(all_results, self.tmpdir)
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir, "checkpoint_comparison.csv")))


# ---------------------------------------------------------------------------
# Integration smoke test
# ---------------------------------------------------------------------------


class TestExperimentRegistry(unittest.TestCase):
    def test_known_experiment(self):
        from src.inference.experiment_registry import get_experiment_meta
        meta = get_experiment_meta("experiment_1")
        self.assertEqual(meta["reference_type"], "analytical")
        self.assertFalse(meta["has_building"])

    def test_unknown_experiment(self):
        from src.inference.experiment_registry import get_experiment_meta
        meta = get_experiment_meta("experiment_999")
        self.assertEqual(meta["domain_type"], "rectangular")


class TestIntegrationSmokeTest(unittest.TestCase):
    """End-to-end test with a tiny model and analytical validation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.tmpdir, "test_config.yaml")
        self.ckpt_dir = os.path.join(self.tmpdir, "checkpoint")
        os.makedirs(self.ckpt_dir)

        # Minimal config
        config = {
            "experiment": {"name": "experiment_1"},
            "data_free": True,
            "training": {
                "learning_rate": 0.001,
                "epochs": 1,
                "batch_size": 4,
                "seed": 42,
            },
            "model": {
                "name": "FourierPINN",
                "width": 8,
                "depth": 2,
                "output_dim": 3,
                "kernel_init": "glorot_uniform",
                "bias_init": 0.0,
                "ff_dims": 16,
                "fourier_scale": 1.0,
            },
            "domain": {"lx": 100.0, "ly": 10.0, "t_final": 100.0},
            "grid": {"nx": 5, "ny": 3, "nt": 3},
            "plotting": {"nx_val": 5, "t_const_val": 50.0, "y_const_plot": 0},
            "physics": {
                "u_const": 0.29,
                "n_manning": 0.03,
                "inflow": None,
                "g": 9.81,
            },
            "loss_weights": {"pde_weight": 1.0, "bc_weight": 1.0, "ic_weight": 1.0, "neg_h_weight": 1.0},
            "device": {"dtype": "float32"},
            "numerics": {"eps": 1e-6},
        }
        with open(self.config_path, "w") as f:
            yaml.dump(config, f)

        # Create a checkpoint by initialising a model
        from src.config import load_config
        from src.training.setup import init_model_from_config
        from flax.core import FrozenDict

        cfg_dict = load_config(self.config_path)
        cfg = FrozenDict(cfg_dict)
        _model, params, _tk, _vk = init_model_from_config(cfg)

        with open(os.path.join(self.ckpt_dir, "model.pkl"), "wb") as f:
            pickle.dump({"params": params, "opt_state": None}, f)

        meta = {"epoch": 1, "best_nse_h": 0.5}
        with open(os.path.join(self.ckpt_dir, "metadata.yaml"), "w") as f:
            yaml.dump(meta, f)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_run_inference(self):
        from src.inference.runner import run_inference

        output_dir = os.path.join(self.tmpdir, "output")
        results = run_inference(
            config_path=self.config_path,
            checkpoint_path=self.ckpt_dir,
            output_dir=output_dir,
            skip_plots=True,
            skip_conservation=True,
        )

        # Check key results exist
        self.assertIn("accuracy", results)
        self.assertIn("nse_h", results["accuracy"])
        self.assertIn("negative_depth", results["accuracy"])
        self.assertIn("temporal_decomposition", results)
        self.assertEqual(results["checkpoint"], "best_nse")
        self.assertGreater(results["n_points"], 0)

        # Check output files
        self.assertTrue(os.path.exists(os.path.join(output_dir, "summary_metrics.yaml")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "predictions.npz")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "report.txt")))

    def test_run_inference_with_conservation(self):
        from src.inference.runner import run_inference

        output_dir = os.path.join(self.tmpdir, "output_cons")
        results = run_inference(
            config_path=self.config_path,
            checkpoint_path=self.ckpt_dir,
            output_dir=output_dir,
            skip_plots=True,
            skip_conservation=False,
        )

        self.assertIn("continuity_residual", results)
        self.assertIn("volume_balance", results)


# Need jax for some test helpers
import jax


# ---------------------------------------------------------------------------
# Min-depth threshold tests
# ---------------------------------------------------------------------------


class TestApplyMinDepth(unittest.TestCase):
    """Unit tests for the _apply_min_depth post-processing function."""

    def test_positive_values_preserved_when_min_depth_zero(self):
        from src.predict.predictor import _apply_min_depth
        preds = jnp.array([[0.001, 0.01, 0.02], [0.1, 0.2, 0.3]])
        result = _apply_min_depth(preds, 0.0)
        np.testing.assert_array_equal(np.array(result), np.array(preds))

    def test_negative_h_clamped_when_min_depth_zero(self):
        from src.predict.predictor import _apply_min_depth
        preds = jnp.array([[-0.05, 0.5, 0.3], [0.1, 0.2, 0.1]])
        result = _apply_min_depth(preds, 0.0)
        expected = jnp.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.1]])
        np.testing.assert_array_almost_equal(np.array(result), np.array(expected))

    def test_zeros_below_threshold(self):
        from src.predict.predictor import _apply_min_depth
        preds = jnp.array([
            [0.001, 0.5, 0.3],   # h=0.001 < 0.005 → dry
            [0.01,  0.2, 0.1],   # h=0.01  >= 0.005 → wet
            [0.004, 0.8, 0.4],   # h=0.004 < 0.005 → dry
            [-0.01, 0.1, 0.2],   # h=-0.01 < 0.005 → dry
        ])
        result = _apply_min_depth(preds, 0.005)
        expected = jnp.array([
            [0.0,  0.0, 0.0],
            [0.01, 0.2, 0.1],
            [0.0,  0.0, 0.0],
            [0.0,  0.0, 0.0],
        ])
        np.testing.assert_array_almost_equal(np.array(result), np.array(expected))

    def test_all_wet(self):
        from src.predict.predictor import _apply_min_depth
        preds = jnp.array([[0.1, 0.5, 0.3], [1.0, 0.2, 0.1]])
        result = _apply_min_depth(preds, 0.005)
        np.testing.assert_array_almost_equal(np.array(result), np.array(preds))

    def test_all_dry(self):
        from src.predict.predictor import _apply_min_depth
        preds = jnp.array([[0.001, 0.5, 0.3], [0.002, 0.2, 0.1]])
        result = _apply_min_depth(preds, 0.005)
        expected = jnp.zeros_like(preds)
        np.testing.assert_array_almost_equal(np.array(result), np.array(expected))

    def test_negative_min_depth_still_clamps_negative_h(self):
        from src.predict.predictor import _apply_min_depth
        preds = jnp.array([[0.001, 0.5, 0.3], [-0.01, 0.1, 0.2]])
        result = _apply_min_depth(preds, -1.0)
        expected = jnp.array([[0.001, 0.5, 0.3], [0.0, 0.0, 0.0]])
        np.testing.assert_array_almost_equal(np.array(result), np.array(expected))


class TestPredictorMinDepth(unittest.TestCase):
    """Tests that Predictor applies min_depth threshold during prediction."""

    def _make_model_and_params(self):
        import flax.linen as nn

        class ConstantModel(nn.Module):
            values: tuple

            @nn.compact
            def __call__(self, x, train=False):
                # Return constant values for every input point
                v = jnp.array(self.values)
                return jnp.broadcast_to(v, (x.shape[0], len(self.values)))

        model = ConstantModel(values=(0.003, 0.5, 0.2))
        variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, 3)))
        params = {"params": variables.get("params", {})}
        return model, params

    def test_predictor_without_min_depth(self):
        from src.predict import Predictor
        model, params = self._make_model_and_params()
        predictor = Predictor(model, batch_size=10, min_depth=0.0)
        coords = jnp.ones((5, 3))
        result = predictor.predict_full(params, coords)
        # h=0.003 should be preserved (no threshold)
        self.assertAlmostEqual(float(result[0, 0]), 0.003, places=5)

    def test_predictor_with_min_depth(self):
        from src.predict import Predictor
        model, params = self._make_model_and_params()
        predictor = Predictor(model, batch_size=10, min_depth=0.005)
        coords = jnp.ones((5, 3))
        result = predictor.predict_full(params, coords)
        # h=0.003 < 0.005 → all zeroed
        np.testing.assert_array_almost_equal(
            np.array(result), np.zeros((5, 3))
        )


if __name__ == "__main__":
    unittest.main()
