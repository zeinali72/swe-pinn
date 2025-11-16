# optimisation/run_sensitivity_analysis.py
"""
Sets up and runs an Optuna SENSITIVITY ANALYSIS study.
This script is for STEP 1 of the HPO process.

Key differences from run_optimization.py:
1.  Uses `optuna.samplers.QMCSampler` for uniform exploration (not exploitation).
2.  Generates a parameter importance plot at the end.
"""
import optuna
import optuna.samplers  # <<<--- IMPORTED SAMPLERS
import optuna.visualization as vis # <<<--- IMPORTED VISUALIZATION
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
    print(f"Added project root to sys.path: {project_root}")

# --- Imports from project ---
try:
    from optimisation.objective_function import objective
    from src.config import load_config
except ImportError as e:
    print("Error: Could not import necessary modules.")
    print(f"Details: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Optuna SENSITIVITY ANALYSIS for SWE-PINN.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the HPO BASE configuration file (e.g., optimisation/configs/hpo_base_fourier.yaml).")
    parser.add_argument("--n_trials", type=int, default=100,
                        help="Number of exploration trials to run (e.g., 100-200 for sensitivity).")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g., sqlite:///study.db). If not provided, defaults to optimisation/database/{study_name}.db")
    parser.add_argument("--study_name", type=str, default="hpo-sensitivity", # <<<--- MODIFIED STUDY NAME
                        help="Name for the Optuna study.")

    args = parser.parse_args()

    # --- Load Base Configuration ---
    try:
        base_config_dict = load_config(args.config)
        print(f"Loaded HPO base configuration from: {args.config}")
    except FileNotFoundError:
        print(f"Error: HPO base config file not found at {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading HPO base config file: {e}")
        sys.exit(1)

    # --- Get HPO Settings from Config ---
    hpo_settings = base_config_dict.get("hpo_settings")
    if not hpo_settings:
        print("Error: 'hpo_settings' section not found in the config file.")
        sys.exit(1)

    data_free_flag = hpo_settings.get("data_free", False)
    enable_gradnorm_flag = hpo_settings.get("enable_gradnorm", False)
    opt_epochs = hpo_settings.get("opt_epochs", 5000)

    print(f"Mode: {'DATA-FREE' if data_free_flag else 'With Data Loss'} (from config)")
    print(f"GradNorm: {'Enabled' if enable_gradnorm_flag else 'Disabled'} (from config)")
    print(f"Sensitivity trials will run for {opt_epochs} epochs each (from config).")

    # --- Update base config dict with explicit opt_epochs for objective function ---
    if "training" not in base_config_dict: base_config_dict["training"] = {}
    base_config_dict["training"]["epochs"] = opt_epochs
    print(f"Sensitivity trials will run for {opt_epochs} epochs each (from config).")

    # --- Setup Optuna Study ---
    if args.storage is None:
        db_dir = os.path.join(project_root, "optimisation", "database")
        os.makedirs(db_dir, exist_ok=True)
        db_file = os.path.join(db_dir, "all_my_studies.db") # Can use one DB for all studies
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

    # --- MODIFICATION: Use QMCSampler for sensitivity analysis ---
    print("Using QMCSampler for uniform exploration (Sensitivity Analysis).")
    sampler = optuna.samplers.QMCSampler()
    # --- END MODIFICATION ---

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_path,
        direction="maximize", # Maximize NSE
        sampler=sampler,      # <<<--- ADDED SAMPLER
        load_if_exists=True,
        pruner=optuna.pruners.NopPruner()  # No pruning during sensitivity analysis
    )

    # Use partial to pass static args AND the determined flags to the objective
    objective_with_config = partial(objective,
                                    base_config_dict=base_config_dict,
                                    data_free=data_free_flag,
                                    enable_gradnorm=enable_gradnorm_flag)

    # --- Run Optimization ---
    print(f"\n--- Starting Optuna SENSITIVITY ANALYSIS ---")
    print(f"Study Name    : {args.study_name}")
    print(f"Storage       : {storage_path}")
    print(f"# Trials      : {args.n_trials}")
    print(f"Objective     : Maximize NSE")
    print(f"Sampler       : QMCSampler (Exploration)")
    print(f"Data-Free Mode: {data_free_flag}")
    print(f"GradNorm Mode : {enable_gradnorm_flag}")
    print(f"Trial Epochs  : {opt_epochs}")
    print(f"Model Name    : {base_config_dict.get('model', {}).get('name', 'N/A')}")
    print(f"Base Config   : {args.config}")
    print("-" * 40)

    start_time_opt = time.time()
    best_trial_config = None

    try:
        study.optimize(objective_with_config, n_trials=args.n_trials, timeout=None, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\nSensitivity analysis interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during sensitivity analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
         total_time_opt = time.time() - start_time_opt
         print(f"\nAnalysis process finished in {total_time_opt:.2f} seconds.")

    # --- Report Best Results & Save Best Config / Plots ---
    print("\n--- Analysis Finished ---")
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
            print("\n--- Best Trial (during exploration) Summary ---")
            print(f"Total analysis time: {total_time_opt:.2f} seconds.")
            print(f"Best Trial Number      : {best_trial.number}")
            best_nse = best_trial.value
            if isinstance(best_nse, (float, int)) and best_nse > -float('inf'):
                 print(f"Best NSE Value         : {best_nse:.6f}")
            else:
                 print(f"Best NSE Value         : {best_nse} (Invalid or not achieved)")

            print("Best Hyperparameters (from exploration):")
            for key, value in sorted(best_trial.params.items()):
                 if value is not None:
                     if isinstance(value, float):
                          print(f"  {key:<25}: {value:.6e}" if abs(value) < 1e-2 or abs(value) > 1e3 else f"  {key:<25}: {value:.6f}")
                     else:
                          print(f"  {key:<25}: {value}")
                 else:
                      print(f"  {key:<25}: Not suggested")
            print("--------------------------")

            # --- Save the best complete configuration (same as before) ---
            save_dir = os.path.join(project_root, "optimisation", "results", args.study_name)
            os.makedirs(save_dir, exist_ok=True)
            
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
                best_trial_config_dict['training'].pop('opt_epochs', None)
                
                save_path_config = os.path.join(save_dir, f"best_trial_{best_trial.number}_config.yaml")
                try:
                    with open(save_path_config, 'w') as f:
                        yaml.dump(best_trial_config_dict, f, default_flow_style=False, sort_keys=False)
                    print(f"\n✅ Best trial's config saved to: {save_path_config}")
                except Exception as e_save:
                    print(f"\n❌ Error saving best configuration: {e_save}")
            else:
                print("\nWarning: Could not find 'full_config' in best trial's user attributes. Configuration not saved.")

            # --- NEW: Generate and save parameter importance plot ---
            try:
                if vis.is_available():
                    print("\nGenerating parameter importance plot...")
                    fig = vis.plot_param_importances(study)
                    save_path_plot = os.path.join(save_dir, "sensitivity_importances.png")
                    fig.write_image(save_path_plot)
                    print(f"✅ Sensitivity analysis plot saved to: {save_path_plot}")
                else:
                    print("\nWarning: `optuna.visualization` not available. Install plotly and kaleido to save plots.")
                    print("         (pip install plotly kaleido)")
            except Exception as e_plot:
                print(f"\n❌ Error generating importance plot: {e_plot}")
            # --- END NEW ---

    except ValueError as e:
        print(f"Could not retrieve best trial: {e}")
    except Exception as e:
        print(f"Error reporting results: {e}")

    print(f"\nStudy results database saved in: {storage_path}")
    print("Consider using 'optuna-dashboard' to visualize results.")


if __name__ == "__main__":
    main()