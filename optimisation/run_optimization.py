# optimisation/run_optimization.py
"""
Sets up and runs the Optuna hyperparameter optimization study.
Saves the full configuration of the best trial.
"""
import optuna
import argparse
import os
import sys
from functools import partial
import time
import yaml
from flax.core import unfreeze, FrozenDict
import numpy as np
import jax.numpy as jnp

# --- Add project root to path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from optimisation.objective_function import objective
    from src.config import load_config
except ImportError as e:
    print("Error: Could not import necessary modules.")
    print(f"Details: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter optimization for SWE-PINN.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the HPO BASE configuration file.")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of optimization trials to run.")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL.")
    parser.add_argument("--study_name", type=str, default="swe-pinn-hpo",
                        help="Name for the Optuna study.")

    args = parser.parse_args()

    # --- Load Base Configuration ---
    try:
        base_config_dict = load_config(args.config)
        print(f"Loaded HPO base configuration from: {args.config}")
    except Exception as e:
        print(f"Error loading HPO base config file: {e}")
        sys.exit(1)

    # --- Get HPO Settings from Config ---
    hpo_settings = base_config_dict.get("hpo_settings", {})
    opt_epochs = hpo_settings.get("opt_epochs", 5000)

    print(f"Mode: DATA-FREE (Physics Only)")
    print(f"GradNorm: Disabled")
    print(f"Optimization trials will run for {opt_epochs} epochs each.")

    # --- Update base config dict with explicit opt_epochs ---
    if "training" not in base_config_dict: base_config_dict["training"] = {}
    base_config_dict["training"]["epochs"] = opt_epochs

    # --- Setup Optuna Study ---
    if args.storage is None:
        db_dir = os.path.join(project_root, "optimisation", "database")
        os.makedirs(db_dir, exist_ok=True)
        db_file = os.path.join(db_dir, "all_my_studies.db")
        storage_path = f"sqlite:///{db_file}"
    else:
        storage_path = args.storage
        if storage_path.startswith("sqlite:///"):
            db_file = storage_path.split("sqlite:///")[-1]
            if not os.path.isabs(db_file):
                db_file = os.path.join(project_root, db_file)
                storage_path = f"sqlite:///{db_file}"
            db_dir = os.path.dirname(db_file)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_path,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=200, interval_steps=50)
    )

    # Use partial to pass static args
    objective_with_config = partial(objective, base_config_dict=base_config_dict)

    # --- Run Optimization ---
    print(f"\n--- Starting Optuna Optimization ---")
    print(f"Study Name    : {args.study_name}")
    print(f"Storage       : {storage_path}")
    print(f"# Trials      : {args.n_trials}")
    print(f"Objective     : Maximize NSE")
    print(f"Trial Epochs  : {opt_epochs}")
    print("-" * 40)

    start_time_opt = time.time()

    try:
        study.optimize(objective_with_config, n_trials=args.n_trials, timeout=None, show_progress_bar=False)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during optimization: {e}")
        import traceback
        traceback.print_exc()
    finally:
         total_time_opt = time.time() - start_time_opt
         print(f"\nOptimization process finished in {total_time_opt:.2f} seconds.")

    # --- Report Best Results & Save Best Config ---
    print("\n--- Optimization Finished ---")
    try:
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        if not complete_trials:
             print("\nNo trials completed successfully.")
        else:
            best_trial = study.best_trial
            print("\n--- Best Trial Summary ---")
            print(f"Best Trial Number      : {best_trial.number}")
            print(f"Best NSE Value         : {best_trial.value}")
            print("Best Hyperparameters:")
            for key, value in sorted(best_trial.params.items()):
                 print(f"  {key:<25}: {value}")
            
            if 'full_config' in best_trial.user_attrs:
                best_trial_config_dict = best_trial.user_attrs['full_config']

                def sanitize_for_yaml(data):
                    if isinstance(data, (jnp.ndarray, np.ndarray)): return data.tolist()
                    if isinstance(data, (jnp.float32, jnp.float64, np.float32, np.float64)): return float(data)
                    if isinstance(data, dict): return {k: sanitize_for_yaml(v) for k, v in data.items()}
                    if isinstance(data, list): return [sanitize_for_yaml(item) for item in data]
                    return data

                if isinstance(best_trial_config_dict, FrozenDict):
                     best_trial_config_dict = unfreeze(best_trial_config_dict)

                best_trial_config_dict = sanitize_for_yaml(best_trial_config_dict)
                best_trial_config_dict.pop('hpo_settings', None)
                best_trial_config_dict['training']['epochs'] = base_config_dict.get('training', {}).get('epochs', 20000)

                save_dir = os.path.join(project_root, "optimisation", "results", args.study_name)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"best_trial_{best_trial.number}_config.yaml")

                with open(save_path, 'w') as f:
                    yaml.dump(best_trial_config_dict, f, default_flow_style=False, sort_keys=False)
                print(f"\n✅ Best trial's configuration saved to: {save_path}")

    except Exception as e:
        print(f"Error reporting results: {e}")


if __name__ == "__main__":
    main()