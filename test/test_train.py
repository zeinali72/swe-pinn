# tests/test_train.py
import os
import shutil
import unittest
from unittest.mock import patch
import yaml

# Add the root directory to the Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.experiment_1.train import main as train_main

class TestTrain(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory for testing."""
        self.test_dir = "test_temp"
        os.makedirs(self.test_dir, exist_ok=True)
        self.config_path = os.path.join(self.test_dir, "test_config.yaml")

    def tearDown(self):
        """Clean up the temporary directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        # Also clean up the results and models directories created during the test
        for dir_name in ["results", "models"]:
            if os.path.exists(dir_name):
                for item in os.listdir(dir_name):
                    if "test_config" in item:
                        item_path = os.path.join(dir_name, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)


    def create_test_config(self):
        """Create a minimal config file for testing."""
        config = {
            'training': {
                'learning_rate': 0.001,
                'epochs': 2,  # Run for only 2 epochs for a quick test
                'batch_size': 4,
                'seed': 42,
                'clip_norm': 1.0,
            },
            'model': {
                'name': "FourierPINN",
                'width': 16,
                'depth': 2,
                'output_dim': 3,
                'kernel_init': "glorot_uniform",
                'bias_init': 0.0,
                'ff_dims': 32,
                'fourier_scale': 1.0,
            },
            'domain': {
                'lx': 1200.0,
                'ly': 100.0,
                't_final': 3600.0,
            },
            'grid': {
                'nx': 10,
                'ny': 5,
                'nt': 5,
            },
            'ic_bc_grid': {
                'nx_ic': 5,
                'ny_ic': 5,
                'ny_bc_left': 5,
                'nt_bc_left': 5,
                'ny_bc_right': 5,
                'nt_bc_right': 5,
                'nx_bc_bottom': 5,
                'nt_bc_other': 5,
                'nx_bc_top': 5,
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
            },
            'sampling': {
                'n_points_pde': 20,
                'n_points_ic': 20,
                'n_points_bc_domain': 20,
            },
            'data_free': True,
            'validation_grid': {
                'n_points_val': 50,
            },
            'wandb': {
                'enable': False,
            },
            'plotting': {
                'nx_val': 20,
                't_const_val': 1800.0,
                'y_const_plot': 0,
            },
            'device': {
                'dtype': "float32",
                'early_stop_min_epochs': 1,
                'early_stop_patience': 1,
            },
            'numerics': {
                'eps': 1e-6,
            }
        }
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)

    @patch('src.training.loop.ask_for_confirmation', return_value=True)
    def test_train_script_runs_without_errors(self, mock_ask_for_confirmation):
        """
        Test that the main training script runs for a few epochs without raising exceptions.
        """
        self.create_test_config()
        try:
            # Run the training script with the test config
            train_main(self.config_path)

            # Check if output directories were created as expected
            # Results/models are scoped under the experiment name directory
            experiment_results_dir = os.path.join("results", "experiment")
            experiment_models_dir = os.path.join("models", "experiment")
            self.assertTrue(os.path.isdir(experiment_results_dir), "Results experiment directory not found.")
            self.assertTrue(os.path.isdir(experiment_models_dir), "Models experiment directory not found.")

            results_dir_content = os.listdir(experiment_results_dir)
            self.assertTrue(any(d.endswith("_test_config") for d in results_dir_content), "Results directory not found.")
            models_dir_content = os.listdir(experiment_models_dir)
            self.assertTrue(any(d.endswith("_test_config") for d in models_dir_content), "Models directory not found.")

            # Find the specific trial directory to check for output files
            trial_name = [d for d in results_dir_content if d.endswith("_test_config")][0]
            self.assertTrue(os.path.exists(os.path.join(experiment_results_dir, trial_name, "final_validation_plot.png")))

            # Check that checkpoint directories exist (dual checkpoint strategy)
            model_trial_dir = os.path.join(experiment_models_dir, trial_name)
            self.assertTrue(os.path.exists(os.path.join(model_trial_dir, "checkpoints", "best_nse")))
            self.assertTrue(os.path.exists(os.path.join(model_trial_dir, "checkpoints", "best_loss")))
            self.assertTrue(os.path.exists(os.path.join(model_trial_dir, "checkpoints", "final")))

            # Verify final checkpoint has model.pkl and metadata.yaml
            final_ckpt_dir = os.path.join(model_trial_dir, "checkpoints", "final")
            self.assertTrue(os.path.exists(os.path.join(final_ckpt_dir, "model.pkl")))
            self.assertTrue(os.path.exists(os.path.join(final_ckpt_dir, "metadata.yaml")))

        except Exception as e:
            self.fail(f"Training script failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main()
