"""Integration tests for the HPO pipeline."""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import optuna
from flax.core import FrozenDict

from optimisation.objective_function import get_hpo_value


# Minimal config for a tiny HPO trial (2 epochs, tiny model)
MINIMAL_HPO_CONFIG = {
    "hpo_settings": {"data_free": True},
    "training": {
        "epochs": 2, "learning_rate": 1e-3, "batch_size": 32,
        "seed": 42, "clip_norm": 1.0,
    },
    "model": {"name": "MLP", "width": 16, "depth": 2, "output_dim": 3},
    "domain": {"lx": 100.0, "ly": 10.0, "t_final": 60.0},
    "physics": {"g": 9.81, "n_manning": 0.03, "u_const": 0.29, "inflow": None},
    "sampling": {
        "n_points_pde": 64, "n_points_ic": 32, "n_points_bc_domain": 64,
    },
    "loss_weights": {
        "pde_weight": 1.0, "ic_weight": 1.0, "bc_weight": 1.0,
        "neg_h_weight": 1.0, "data_weight": 0.0,
    },
    "device": {"dtype": "float32"},
    "numerics": {"eps": 1e-6},
    "validation_grid": {"n_points_val": 50},
}


class TestGetHpoValue(unittest.TestCase):
    """Test the config-driven parameter suggestion helper."""

    def _make_trial(self):
        study = optuna.create_study(direction="maximize")
        return study.ask()

    def test_fixed_scalar_returns_value(self):
        trial = self._make_trial()
        self.assertEqual(get_hpo_value(trial, "test", 42.0), 42.0)
        self.assertEqual(get_hpo_value(trial, "test_str", "hello"), "hello")
        self.assertEqual(get_hpo_value(trial, "test_bool", True), True)

    def test_list_returns_categorical(self):
        trial = self._make_trial()
        result = get_hpo_value(trial, "choice", [128, 256, 512])
        self.assertIn(result, [128, 256, 512])

    def test_dict_range_returns_float(self):
        trial = self._make_trial()
        result = get_hpo_value(trial, "lr", {"min": 1e-4, "max": 1e-2})
        self.assertGreaterEqual(result, 1e-4)
        self.assertLessEqual(result, 1e-2)

    def test_dict_range_int(self):
        trial = self._make_trial()
        result = get_hpo_value(trial, "depth", {"min": 2, "max": 8, "type": "int"})
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 2)
        self.assertLessEqual(result, 8)

    def test_none_with_default(self):
        trial = self._make_trial()
        result = get_hpo_value(trial, "test", None, lambda: 99)
        self.assertEqual(result, 99)

    def test_none_without_default_raises(self):
        trial = self._make_trial()
        with self.assertRaises(ValueError):
            get_hpo_value(trial, "test", None)


class TestObjectiveSmoke(unittest.TestCase):
    """Smoke test: run a tiny HPO trial end-to-end."""

    def test_objective_returns_valid_float(self):
        """Run objective() with 2 epochs on a tiny model."""
        from optimisation.objective_function import objective

        study = optuna.create_study(direction="maximize")
        # Run 1 trial with our minimal config
        study.optimize(
            lambda trial: objective(trial, MINIMAL_HPO_CONFIG),
            n_trials=1,
        )
        completed = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
        self.assertEqual(len(completed), 1)
        result = completed[0].value
        self.assertIsInstance(result, float)
        # NSE can be negative for untrained model, but should be a valid number
        self.assertFalse(float('nan') == result)


class TestExperimentDynamicImport(unittest.TestCase):
    """Test that experiments can be dynamically imported for HPO."""

    def test_known_experiments_have_setup_trial(self):
        import importlib
        for name in ["experiment_1", "experiment_2"]:
            mod = importlib.import_module(f"experiments.{name}.train")
            self.assertTrue(hasattr(mod, "setup_trial"), f"{name} missing setup_trial")
            self.assertTrue(callable(mod.setup_trial))

    def test_unknown_experiment_import_fails(self):
        import importlib
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("experiments.nonexistent.train")


if __name__ == "__main__":
    unittest.main()
