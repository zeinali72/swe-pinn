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

# --- Add project root to path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from optimisation.objective_function import objective
    from optimisation.utils import sanitize_for_yaml, setup_study_storage, create_storage, get_direction
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
    parser.add_argument("--fresh", action="store_true",
                        help="Delete existing study and start fresh (overrides --resume).")
    parser.add_argument("--resume", action="store_true",
                        help="Explicitly continue an existing study (default behavior; opposite of --fresh).")

    args = parser.parse_args()

    # --- Load Base Configuration ---
    try:
        base_config_dict = load_config(args.config)
        print(f"Loaded HPO config: {args.config}")
    except Exception as e:
        print(f"Error loading HPO base config file: {e}")
        sys.exit(1)

    # --- Get HPO Settings from Config ---
    hpo_settings = base_config_dict.get("hpo_settings", {})
    opt_epochs = base_config_dict.get("training", {}).get("epochs", 5000)
    data_free = hpo_settings.get("data_free", True)
    objective_key = hpo_settings.get("objective_key", "nse_h")
    direction = get_direction(objective_key)

    # --- Setup Storage ---
    storage_backend = hpo_settings.get("storage_backend", "local")
    storage_url = setup_study_storage(storage_backend, project_root, cli_storage=args.storage)
    storage = create_storage(storage_url)

    study_name = args.study_name if args.study_name != "swe-pinn-hpo" else \
        hpo_settings.get("study_name", args.study_name)

    # --- Fresh start: delete old study if requested ---
    if args.fresh:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f"Deleted existing study '{study_name}'.")
        except KeyError:
            pass  # Study didn't exist

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=200, interval_steps=50)
    )

    existing = len(study.trials)

    # Use partial to pass static args
    objective_with_config = partial(objective, base_config_dict=base_config_dict)

    # --- Run Optimization ---
    mode = "DATA-FREE" if data_free else "DATA-DRIVEN"
    print(f"\n--- Optuna HPO ---")
    print(f"Study         : {study_name} ({'continuing' if existing else 'new'})")
    print(f"Existing      : {existing} trials")
    print(f"New trials    : {args.n_trials}")
    print(f"Objective     : {direction} {objective_key}")
    print(f"Mode          : {mode}")
    print(f"Epochs/trial  : {opt_epochs}")
    print(f"Storage       : {storage_url}")
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
         print(f"\nFinished in {total_time_opt:.2f}s.")

    # --- Report Best Results & Save Best Config ---
    try:
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        if not complete_trials:
             print("\nNo trials completed successfully.")
        else:
            best_trial = study.best_trial
            print(f"\n--- Best Trial ---")
            print(f"Trial #{best_trial.number}  {objective_key} = {best_trial.value}")
            for key, value in sorted(best_trial.params.items()):
                 print(f"  {key:<25}: {value}")

            if 'full_config' in best_trial.user_attrs:
                best_trial_config_dict = best_trial.user_attrs['full_config']

                if isinstance(best_trial_config_dict, FrozenDict):
                     best_trial_config_dict = unfreeze(best_trial_config_dict)

                best_trial_config_dict = sanitize_for_yaml(best_trial_config_dict)
                best_trial_config_dict.pop('hpo_settings', None)
                best_trial_config_dict.pop('hpo_hyperparameters', None)

                save_dir = os.path.join(project_root, "optimisation", "results", study_name)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"best_trial_{best_trial.number}_config.yaml")

                with open(save_path, 'w') as f:
                    yaml.dump(best_trial_config_dict, f, default_flow_style=False, sort_keys=False)
                print(f"Config saved: {save_path}")

    except Exception as e:
        print(f"Error reporting results: {e}")


if __name__ == "__main__":
    main()
