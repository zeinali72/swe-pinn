# optimisation/run_optimization.py
"""
Sets up and runs the Optuna hyperparameter optimization study.
"""
import optuna
import argparse
import os
import sys
from functools import partial
import time # For summary timing

# --- Add project root to path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to sys.path: {project_root}")

# --- Imports from project ---
try:
    from optimisation.objective_function import objective
    from src.config import load_config
    # Removed print_final_summary import from src.reporting, will format directly
except ImportError as e:
    print("Error: Could not import necessary modules.")
    print("Ensure 'objective_function.py' and 'optimization_train_loop.py' are in the 'optimisation' directory.")
    print("Ensure you are running this script from the project root directory containing 'src' and 'optimisation'.")
    print(f"Details: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter optimization for SWE-PINN.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the BASE configuration file (e.g., experiments/one_building_config_gradnorm.yaml).")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of optimization trials to run.")
    parser.add_argument("--storage", type=str, default="sqlite:///optimisation_study.db", # Default DB name
                        help="Optuna storage URL (e.g., sqlite:///study.db).")
    parser.add_argument("--study_name", type=str, default="swe-pinn-hpo", # Default study name
                        help="Name for the Optuna study.")
    parser.add_argument("--data_free", action='store_true',
                        help="Force optimization into data-free mode (ignores data_weight from config).")
    parser.add_argument("--opt_epochs", type=int, default=5000,
                        help="Number of epochs to run each optimization trial.")

    args = parser.parse_args()

    # --- Load Base Configuration ---
    try:
        # Load config and ensure DTYPE/EPS are set globally for trial functions
        base_config_dict = load_config(args.config)
        print(f"Loaded base configuration from: {args.config}")
    except FileNotFoundError:
        print(f"Error: Base config file not found at {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading base config file: {e}")
        sys.exit(1)

    # --- Determine Data-Free Flag (Argument takes precedence) ---
    data_free_flag = args.data_free
    if data_free_flag:
        print("Mode: DATA-FREE optimization enforced via --data_free argument.")
    else:
        # Check config only if argument not used
        data_weight_in_config = base_config_dict.get("loss_weights", {}).get("data_weight", 0.0)
        if data_weight_in_config <= 0:
            print("Mode: Data weight is <= 0 or missing in config. Running DATA-FREE.")
            data_free_flag = True # Treat as data-free if weight is non-positive
        else:
            print("Mode: Running WITH data loss (data_weight > 0 in config and --data_free not specified).")
            data_free_flag = False

    # --- Determine GradNorm Flag (Based *only* on config) ---
    enable_gradnorm_flag = base_config_dict.get("gradnorm", {}).get("enable", False)
    print(f"GradNorm (from config): {enable_gradnorm_flag}")
    # The objective function will handle the interaction between these two flags.

    # --- Add/Override Optimization Epochs ---
    if "training" not in base_config_dict: base_config_dict["training"] = {}
    base_config_dict["training"]["opt_epochs"] = args.opt_epochs
    print(f"Optimization trials will run for {args.opt_epochs} epochs each.")

    # --- Setup Optuna Study ---
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
        direction="maximize", # Maximize NSE
        load_if_exists=True,
        # Increased warmup/interval for potentially longer epochs
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10, interval_steps=10)
    )

    # Use partial to pass static args AND the determined flags to the objective
    objective_with_config = partial(objective,
                                    base_config_dict=base_config_dict,
                                    data_free=data_free_flag,
                                    enable_gradnorm=enable_gradnorm_flag)

    # --- Run Optimization ---
    print(f"\n--- Starting Optuna Optimization ---")
    print(f"Study Name    : {args.study_name}")
    print(f"Storage       : {storage_path}")
    print(f"# Trials      : {args.n_trials}")
    print(f"Objective     : Maximize NSE")
    print(f"Data-Free Mode: {data_free_flag}")      # Report the actual mode being used
    print(f"GradNorm Mode : {enable_gradnorm_flag}") # Report the actual mode being used
    print(f"Trial Epochs  : {args.opt_epochs}")
    print(f"Model Name    : {base_config_dict.get('model', {}).get('name', 'N/A')}")
    print(f"Base Config   : {args.config}")
    print("-" * 40)

    start_time_opt = time.time()
    try:
        study.optimize(objective_with_config, n_trials=args.n_trials, timeout=None, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during optimization: {e}")
        import traceback
        traceback.print_exc()
    finally:
         total_time_opt = time.time() - start_time_opt
         print(f"\nOptimization process finished in {total_time_opt:.2f} seconds.")

    # --- Report Best Results ---
    print("\n--- Optimization Finished ---")
    try:
        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        fail_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.FAIL])

        print(f"Study statistics: ")
        print(f"  Number of finished trials: {len(study.trials)}")
        print(f"  Number of pruned trials  : {len(pruned_trials)}")
        print(f"  Number of complete trials: {len(complete_trials)}")
        print(f"  Number of failed trials  : {len(fail_trials)}")

        if not complete_trials:
             print("\nNo trials completed successfully.")
        else:
            best_trial = study.best_trial
            print("\n--- Best Trial Summary ---")
            print(f"Total optimization time: {total_time_opt:.2f} seconds.")
            print(f"Best Trial Number      : {best_trial.number}")
            best_nse = best_trial.value
            if isinstance(best_nse, (float, int)) and best_nse > -float('inf'):
                 print(f"Best NSE Value         : {best_nse:.6f}")
            else:
                 print(f"Best NSE Value         : {best_nse} (Invalid or not achieved)")

            print("Best Hyperparameters:")
            for key, value in sorted(best_trial.params.items()):
                 # Check if the parameter exists (it might be None if not suggested in that trial branch)
                 if value is not None:
                     if isinstance(value, float):
                          print(f"  {key:<25}: {value:.6e}" if abs(value) < 1e-2 or abs(value) > 1e3 else f"  {key:<25}: {value:.6f}")
                     else:
                          print(f"  {key:<25}: {value}")
                 else:
                      print(f"  {key:<25}: Not suggested (expected)") # Indicate params not used in this branch

            print("--------------------------")

    except ValueError as e:
        print(f"Could not retrieve best trial: {e}")
    except Exception as e:
        print(f"Error reporting results: {e}")

    print(f"\nStudy results saved in: {storage_path}")
    print("Consider using 'optuna-dashboard' to visualize results.")


if __name__ == "__main__":
    main()