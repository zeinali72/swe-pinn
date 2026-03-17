# test/test_setup_trial.py
"""Integration tests for the setup_trial() interface across all experiments.

Verifies that each experiment's setup_trial(cfg_dict) returns a dict with
all required keys, that validation_fn produces metrics, and that the HPO
wrapper handles all edge cases.
"""
import os
import sys
import unittest

os.environ["JAX_PLATFORM_NAME"] = "cpu"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from src.config import load_config


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


def _load_experiment_config(experiment_num, **overrides):
    """Load the real experiment config and apply test overrides."""
    cfg_dict = load_config(f"configs/experiment_{experiment_num}.yaml")
    # Shrink for fast testing
    cfg_dict["training"]["epochs"] = 2
    cfg_dict["training"]["batch_size"] = 4
    cfg_dict["model"]["width"] = 8
    cfg_dict["model"]["depth"] = 2
    if "ff_dims" in cfg_dict.get("model", {}):
        cfg_dict["model"]["ff_dims"] = 8
    # Use small sampling counts
    sampling = cfg_dict.setdefault("sampling", {})
    for key in list(sampling.keys()):
        if key.startswith("n_points_"):
            sampling[key] = 20
    # Disable Aim
    cfg_dict["aim"] = {"enable": False}
    cfg_dict.update(overrides)
    return cfg_dict


def _assert_setup_trial_interface(test_case, ctx):
    """Assert the setup_trial() return dict has the required interface."""
    missing = REQUIRED_KEYS - set(ctx.keys())
    test_case.assertEqual(missing, set(), f"Missing keys: {missing}")
    test_case.assertGreater(ctx["num_batches"], 0)
    test_case.assertIsNotNone(ctx["model"])
    test_case.assertIsNotNone(ctx["params"])
    test_case.assertTrue(callable(ctx["validation_fn"]))
    test_case.assertTrue(callable(ctx["generate_epoch_data_jit"]))
    test_case.assertTrue(callable(ctx["scan_body"]))
    test_case.assertTrue(callable(ctx["compute_all_losses_fn"]))


def _assert_validation_fn_output(test_case, ctx):
    """Assert validation_fn returns a dict with at least nse_h."""
    metrics = ctx["validation_fn"](ctx["model"], ctx["params"])
    test_case.assertIsInstance(metrics, dict)
    test_case.assertIn("nse_h", metrics)


def _assert_epoch_data_runnable(test_case, ctx):
    """Assert generate_epoch_data_jit produces a dict and scan runs."""
    key = jax.random.PRNGKey(0)
    batch = ctx["generate_epoch_data_jit"](key)
    test_case.assertIsInstance(batch, dict)


# ─── Experiment 1: Analytical dam-break ──────────────────────────────────────

class TestSetupTrialExperiment1(unittest.TestCase):
    """Test setup_trial() for Experiment 1 (analytical dam-break)."""

    def test_returns_required_keys(self):
        from experiments.experiment_1.train import setup_trial
        ctx = setup_trial(_make_base_config())
        _assert_setup_trial_interface(self, ctx)

    def test_validation_fn_returns_metrics(self):
        from experiments.experiment_1.train import setup_trial
        ctx = setup_trial(_make_base_config())
        metrics = ctx["validation_fn"](ctx["model"], ctx["params"])
        self.assertIn("nse_h", metrics)
        self.assertIn("rmse_h", metrics)
        # Experiment 1 also computes hu/hv analytically
        self.assertIn("nse_hu", metrics)
        self.assertIn("nse_hv", metrics)

    def test_zero_batch_size_returns_sentinel(self):
        from experiments.experiment_1.train import setup_trial
        cfg_dict = _make_base_config()
        cfg_dict["training"]["batch_size"] = 999999
        ctx = setup_trial(cfg_dict)
        self.assertEqual(ctx.get("num_batches", 0), 0)

    def test_generate_epoch_data_callable(self):
        from experiments.experiment_1.train import setup_trial
        ctx = setup_trial(_make_base_config())
        _assert_epoch_data_runnable(self, ctx)


# ─── Experiment 2: Building obstacle ─────────────────────────────────────────

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
        _assert_setup_trial_interface(self, ctx)

    def test_validation_fn_returns_metrics(self):
        from experiments.experiment_2.train import setup_trial
        cfg_dict = _make_base_config(
            scenario='experiment_2',
            building={'x_min': 30.0, 'x_max': 40.0, 'y_min': 3.0, 'y_max': 7.0},
        )
        cfg_dict['loss_weights']['building_bc_weight'] = 1.0
        cfg_dict['sampling']['n_points_bc_building'] = 20
        ctx = setup_trial(cfg_dict)
        _assert_validation_fn_output(self, ctx)


# ─── Experiment 3: Terrain slope x-direction ─────────────────────────────────

class TestSetupTrialExperiment3(unittest.TestCase):
    """Test setup_trial() for Experiment 3 (terrain, x-direction slope)."""

    def test_returns_required_keys(self):
        from experiments.experiment_3.train import setup_trial
        ctx = setup_trial(_load_experiment_config(3))
        _assert_setup_trial_interface(self, ctx)

    def test_validation_fn_returns_metrics(self):
        from experiments.experiment_3.train import setup_trial
        ctx = setup_trial(_load_experiment_config(3))
        _assert_validation_fn_output(self, ctx)

    def test_generate_epoch_data_callable(self):
        from experiments.experiment_3.train import setup_trial
        ctx = setup_trial(_load_experiment_config(3))
        _assert_epoch_data_runnable(self, ctx)


# ─── Experiment 4: Terrain slope x+y ─────────────────────────────────────────

class TestSetupTrialExperiment4(unittest.TestCase):
    """Test setup_trial() for Experiment 4 (terrain, x+y slope, split inflow)."""

    def test_returns_required_keys(self):
        from experiments.experiment_4.train import setup_trial
        ctx = setup_trial(_load_experiment_config(4))
        _assert_setup_trial_interface(self, ctx)

    def test_validation_fn_returns_metrics(self):
        from experiments.experiment_4.train import setup_trial
        ctx = setup_trial(_load_experiment_config(4))
        _assert_validation_fn_output(self, ctx)


# ─── Experiment 5: Synthetic complexity stage 1 ──────────────────────────────

class TestSetupTrialExperiment5(unittest.TestCase):
    """Test setup_trial() for Experiment 5 (single left inflow)."""

    def test_returns_required_keys(self):
        from experiments.experiment_5.train import setup_trial
        ctx = setup_trial(_load_experiment_config(5))
        _assert_setup_trial_interface(self, ctx)

    def test_validation_fn_returns_metrics(self):
        from experiments.experiment_5.train import setup_trial
        ctx = setup_trial(_load_experiment_config(5))
        _assert_validation_fn_output(self, ctx)


# ─── Experiment 6: Synthetic complexity stage 2 ──────────────────────────────

class TestSetupTrialExperiment6(unittest.TestCase):
    """Test setup_trial() for Experiment 6 (split inflow + left wall)."""

    def test_returns_required_keys(self):
        from experiments.experiment_6.train import setup_trial
        ctx = setup_trial(_load_experiment_config(6))
        _assert_setup_trial_interface(self, ctx)

    def test_validation_fn_returns_metrics(self):
        from experiments.experiment_6.train import setup_trial
        ctx = setup_trial(_load_experiment_config(6))
        _assert_validation_fn_output(self, ctx)


# ─── Experiment 7: Irregular domain ──────────────────────────────────────────

class TestSetupTrialExperiment7(unittest.TestCase):
    """Test setup_trial() for Experiment 7 (irregular boundaries, mesh-based)."""

    def test_returns_required_keys(self):
        from experiments.experiment_7.train import setup_trial
        ctx = setup_trial(_load_experiment_config(7))
        _assert_setup_trial_interface(self, ctx)
        # Experiment 7 should also populate domain bounds
        self.assertIn("x_min", ctx["cfg"]["domain"])

    def test_validation_fn_returns_metrics(self):
        from experiments.experiment_7.train import setup_trial
        ctx = setup_trial(_load_experiment_config(7))
        _assert_validation_fn_output(self, ctx)

    def test_generate_epoch_data_callable(self):
        from experiments.experiment_7.train import setup_trial
        ctx = setup_trial(_load_experiment_config(7))
        _assert_epoch_data_runnable(self, ctx)


# ─── Experiment 8: Real urban domain ─────────────────────────────────────────

class TestSetupTrialExperiment8(unittest.TestCase):
    """Test setup_trial() for Experiment 8 (real urban, combined NSE)."""

    def test_returns_required_keys(self):
        from experiments.experiment_8.train import setup_trial
        ctx = setup_trial(_load_experiment_config(8))
        _assert_setup_trial_interface(self, ctx)

    def test_validation_fn_returns_combined_nse(self):
        """Experiment 8 validation must return selection_metric (combined NSE)."""
        from experiments.experiment_8.train import setup_trial
        ctx = setup_trial(_load_experiment_config(8))
        metrics = ctx["validation_fn"](ctx["model"], ctx["params"])
        self.assertIn("selection_metric", metrics)
        self.assertIn("nse_h", metrics)
        self.assertIn("nse_hu", metrics)
        self.assertIn("nse_hv", metrics)
        self.assertIn("combined_nse", metrics)

    def test_generate_epoch_data_callable(self):
        from experiments.experiment_8.train import setup_trial
        ctx = setup_trial(_load_experiment_config(8))
        _assert_epoch_data_runnable(self, ctx)


# ─── HPO Wrapper Tests ───────────────────────────────────────────────────────

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
        """HPO wrapper should complete epochs and return a finite metric."""
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
        self.assertIsInstance(result, float)

    def test_invalid_scenario_returns_negative(self):
        """HPO wrapper should return -1.0 for nonexistent scenario."""
        import optuna
        from optimisation.optimization_train_loop import run_training_trial

        cfg_dict = _make_base_config()
        cfg_dict["scenario"] = "experiment_99"
        cfg_dict["training"]["epochs"] = 1

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        result = run_training_trial(trial, FrozenDict(cfg_dict))
        self.assertEqual(result, -1.0)

    def test_nonexistent_objective_key_returns_negative(self):
        """HPO wrapper should return -1.0 when objective_key not in metrics."""
        import optuna
        from optimisation.optimization_train_loop import run_training_trial

        cfg_dict = _make_base_config()
        cfg_dict["scenario"] = "experiment_1"
        cfg_dict["training"]["epochs"] = 1
        cfg_dict["hpo_settings"] = {"objective_key": "nonexistent_metric"}

        study = optuna.create_study(direction="maximize")
        trial = study.ask()
        result = run_training_trial(trial, FrozenDict(cfg_dict))
        # objective_key missing from metrics → current stays at -inf → returns -1.0
        self.assertEqual(result, -1.0)

    def test_hpo_with_experiment_2_building(self):
        """HPO wrapper should work with experiment 2 building scenario."""
        import optuna
        from optimisation.optimization_train_loop import run_training_trial

        cfg_dict = _make_base_config(
            scenario='experiment_2',
            building={'x_min': 30.0, 'x_max': 40.0, 'y_min': 3.0, 'y_max': 7.0},
        )
        cfg_dict['loss_weights']['building_bc_weight'] = 1.0
        cfg_dict['sampling']['n_points_bc_building'] = 20
        cfg_dict["training"]["epochs"] = 2
        cfg_dict["hpo_settings"] = {"objective_key": "rmse_h"}

        study = optuna.create_study(direction="minimize")
        trial = study.ask()
        result = run_training_trial(trial, FrozenDict(cfg_dict))
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0.0)


if __name__ == '__main__':
    unittest.main()
