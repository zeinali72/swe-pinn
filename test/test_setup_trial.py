# test/test_setup_trial.py
"""Integration tests for the setup_trial() interface across all experiments.

Verifies that each experiment's setup_trial(cfg_dict) returns a dict with
all required keys, and that the HPO wrapper can handle the result.
"""
import os
import sys
import unittest

os.environ["JAX_PLATFORM_NAME"] = "cpu"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from flax.core import FrozenDict


# Keys that every setup_trial() must return
REQUIRED_KEYS = {
    "cfg", "cfg_dict", "model", "params", "train_key",
    "optimiser", "opt_state", "generate_epoch_data_jit",
    "scan_body", "num_batches", "validation_fn", "data_free",
    "compute_all_losses_fn", "experiment_name",
}


def _make_base_config(**overrides):
    """Create a minimal config dict for testing setup_trial()."""
    cfg = {
        'training': {
            'learning_rate': 0.001,
            'epochs': 2,
            'batch_size': 4,
            'seed': 42,
            'clip_norm': 1.0,
        },
        'model': {
            'name': "MLP",
            'width': 8,
            'depth': 2,
            'output_dim': 3,
        },
        'domain': {
            'lx': 100.0,
            'ly': 10.0,
            't_final': 60.0,
        },
        'physics': {
            'u_const': 0.29,
            'n_manning': 0.03,
            'inflow': None,
            'g': 9.81,
        },
        'loss_weights': {
            'pde_weight': 1.0,
            'bc_weight': 1.0,
            'ic_weight': 1.0,
            'neg_h_weight': 1.0,
        },
        'sampling': {
            'n_points_pde': 20,
            'n_points_ic': 20,
            'n_points_bc_domain': 20,
        },
        'data_free': True,
        'validation_grid': {
            'n_points_val': 10,
        },
        'device': {
            'dtype': "float32",
        },
        'numerics': {
            'eps': 1e-6,
        },
    }
    cfg.update(overrides)
    return cfg


class TestSetupTrialExperiment1(unittest.TestCase):
    """Test setup_trial() for Experiment 1 (analytical dam-break)."""

    def test_returns_required_keys(self):
        from experiments.experiment_1.train import setup_trial
        cfg_dict = _make_base_config()
        ctx = setup_trial(cfg_dict)
        missing = REQUIRED_KEYS - set(ctx.keys())
        self.assertEqual(missing, set(), f"Missing keys: {missing}")

    def test_validation_fn_returns_metrics(self):
        from experiments.experiment_1.train import setup_trial
        cfg_dict = _make_base_config()
        ctx = setup_trial(cfg_dict)
        metrics = ctx["validation_fn"](ctx["model"], ctx["params"])
        self.assertIn("nse_h", metrics)
        self.assertIn("rmse_h", metrics)

    def test_zero_batch_size_returns_sentinel(self):
        from experiments.experiment_1.train import setup_trial
        cfg_dict = _make_base_config()
        cfg_dict["training"]["batch_size"] = 999999
        ctx = setup_trial(cfg_dict)
        self.assertEqual(ctx.get("num_batches", 0), 0)

    def test_generate_epoch_data_jit_callable(self):
        import jax
        from experiments.experiment_1.train import setup_trial
        cfg_dict = _make_base_config()
        ctx = setup_trial(cfg_dict)
        key = jax.random.PRNGKey(0)
        batch = ctx["generate_epoch_data_jit"](key)
        self.assertIsInstance(batch, dict)


class TestSetupTrialExperiment2(unittest.TestCase):
    """Test setup_trial() for Experiment 2 (building obstacle)."""

    def test_returns_required_keys(self):
        from experiments.experiment_2.train import setup_trial
        cfg_dict = _make_base_config(
            scenario='experiment_2',
            building={'x_min': 30.0, 'x_max': 40.0, 'y_min': 3.0, 'y_max': 7.0},
        )
        cfg_dict['loss_weights']['building_bc_weight'] = 1.0
        cfg_dict['sampling']['n_points_bc_building'] = 20
        ctx = setup_trial(cfg_dict)
        missing = REQUIRED_KEYS - set(ctx.keys())
        self.assertEqual(missing, set(), f"Missing keys: {missing}")


class TestSetupTrialHPOWrapper(unittest.TestCase):
    """Test that the HPO wrapper handles setup_trial() output correctly."""

    def test_handles_zero_num_batches(self):
        """HPO wrapper should return -1.0 when num_batches is 0."""
        import optuna
        from optimisation.optimization_train_loop import run_training_trial

        cfg_dict = _make_base_config()
        cfg_dict["training"]["batch_size"] = 999999
        cfg_dict["scenario"] = "experiment_1"
        cfg_dict["training"]["epochs"] = 1

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        result = run_training_trial(trial, FrozenDict(cfg_dict))
        self.assertEqual(result, -1.0)

    def test_runs_one_epoch(self):
        """HPO wrapper should complete 1 epoch and return a finite metric."""
        import optuna
        from optimisation.optimization_train_loop import run_training_trial

        cfg_dict = _make_base_config()
        cfg_dict["scenario"] = "experiment_1"
        cfg_dict["training"]["epochs"] = 2
        # Use rmse_h as objective since nse_h can be -inf with tiny models
        cfg_dict["hpo_settings"] = {"objective_key": "rmse_h"}

        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        result = run_training_trial(trial, FrozenDict(cfg_dict))
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)  # RMSE is always positive

    def test_objective_key_from_config(self):
        """HPO wrapper should use objective_key from hpo_settings."""
        import optuna
        from optimisation.optimization_train_loop import run_training_trial

        cfg_dict = _make_base_config()
        cfg_dict["scenario"] = "experiment_1"
        cfg_dict["training"]["epochs"] = 1
        cfg_dict["hpo_settings"] = {"objective_key": "rmse_h"}

        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        result = run_training_trial(trial, FrozenDict(cfg_dict))
        # Should succeed — rmse_h is returned by experiment 1's validation_fn
        self.assertIsInstance(result, float)


if __name__ == '__main__':
    unittest.main()
