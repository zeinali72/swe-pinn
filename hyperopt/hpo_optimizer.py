# /workspaces/swe-pinn/hyperopt/hpo_optimizer.py
import os
import yaml
import optuna
from optuna.trial import TrialState
import time
import sys
import shutil
import numpy as np
import jax.numpy as jnp

# --- Ensure src and hyperopt modules can be imported ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# <<<--- Use the MODIFIED hpo_train --- >>>
from hyperopt.hpo_train import run_hpo_trial_with_nse # Import NSE trial runner
from src.config import load_config, DTYPE # Load base config, set DTYPE

# --- Configuration ---
BASE_CONFIG_PATH = "experiments/one_building_config.yaml" # Use config with building for NSE
N_TRIALS = 50  # Number of optimization trials
STUDY_NAME = "swe-pinn-hpo-nse-objective-v3" # Choose a descriptive name
STORAGE_PATH = f"sqlite:///hyperopt/{STUDY_NAME}.db" # Database file path

# --- HPO Specific Training Settings ---
# These epochs determine how long each trial runs before NSE is evaluated
HPO_EPOCHS = 5000 # Number of epochs per HPO trial
# Removed HPO specific early stopping vars, as the logic was removed from hpo_train

# --- Objective Function for Optuna ---
def objective(trial: optuna.trial.Trial) -> float:
    """
    Defines a single optimization trial targeting NSE maximization using the modified hpo_train.
    """
    # 1. Load the base configuration
    try:
        config = load_config(BASE_CONFIG_PATH) # Sets DTYPE globally
        config = dict(config) # Convert to mutable dict
    except FileNotFoundError:
        print(f"ERROR: Base config file not found at {BASE_CONFIG_PATH}")
        return -float('inf') # Indicate failure for maximization

    # --- 2. Suggest Hyperparameters (Practical Ranges) ---
    config['training']['learning_rate'] = trial.suggest_float("learning_rate", 5e-5, 5e-3, log=True)
    config['training']['batch_size'] = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])

    # Model (assuming FourierPINN from base config)
    if config['model']['name'] == "FourierPINN":
        config['model']['width'] = trial.suggest_categorical("model_width", [128, 256, 512])
        config['model']['depth'] = trial.suggest_int("model_depth", 3, 6) # Slightly reduced max depth range
        config['model']['ff_dims'] = trial.suggest_categorical("ff_dims", [256, 512]) # Reduced ff_dims range
        config['model']['fourier_scale'] = trial.suggest_float("fourier_scale", 8.0, 20.0) # Adjusted range

    # Loss weights (Excluding data_weight for HPO training step, but keep it in config if needed later)
    config['loss_weights']['pde_weight'] = trial.suggest_float("pde_weight", 1e3, 1e6, log=True) # Adjusted PDE weight
    config['loss_weights']['bc_weight'] = trial.suggest_float("bc_weight", 1e1, 1e3, log=True) # Adjusted BC weight
    config['loss_weights']['ic_weight'] = trial.suggest_float("ic_weight", 1e1, 1e3, log=True) # Adjusted IC weight
    # Keep building weight as we are using the building config
    config['loss_weights']['building_bc_weight'] = trial.suggest_float("building_bc_weight", 1e1, 1e4, log=True) # Adjusted building weight
    # Keep data_weight in config dictionary if present, but it won't be used unless
    # training_dataset_sample.npy exists and data_points_full is loaded in hpo_train.py
    # If you *never* want to use data loss during HPO, uncomment the next line:
    # config['loss_weights'].pop('data_weight', None)

    # Grid Resolution params (Suggest directly)
    config['grid']['nx'] = trial.suggest_int("grid_nx", 40, 80)
    config['grid']['ny'] = trial.suggest_int("grid_ny", 15, 40)
    config['grid']['nt'] = trial.suggest_int("grid_nt", 15, 40)

    # IC/BC Grid Points
    config['ic_bc_grid']['nx_ic'] = trial.suggest_int("nx_ic", 30, 70)
    config['ic_bc_grid']['ny_ic'] = trial.suggest_int("ny_ic", 30, 70)
    config['ic_bc_grid']['ny_bc_left'] = trial.suggest_int("ny_bc_left", 20, 40)
    config['ic_bc_grid']['nt_bc_left'] = trial.suggest_int("nt_bc_left", 15, 30)
    config['ic_bc_grid']['ny_bc_right'] = trial.suggest_int("ny_bc_right", 20, 40)
    config['ic_bc_grid']['nt_bc_right'] = trial.suggest_int("nt_bc_right", 15, 30)
    config['ic_bc_grid']['nx_bc_bottom'] = trial.suggest_int("nx_bc_bottom", 30, 70)
    config['ic_bc_grid']['nt_bc_other'] = trial.suggest_int("nt_bc_other", 15, 30)
    config['ic_bc_grid']['nx_bc_top'] = trial.suggest_int("nx_bc_top", 30, 70)

    # Building Grid Points
    config['building']['nx'] = trial.suggest_int("building_nx", 15, 30)
    config['building']['ny'] = trial.suggest_int("building_ny", 15, 30)
    config['building']['nt'] = trial.suggest_int("building_nt", 15, 30)

    # Set HPO training epochs
    config['training']['epochs'] = HPO_EPOCHS
    # Remove early stopping keys as they are not used in the modified hpo_train
    config['device'].pop('early_stop_min_epochs', None)
    config['device'].pop('early_stop_patience', None)

    # --- 3. Run the training trial using the modified function ---
    trial_nse = -float('inf')
    try:
        print(f"\n--- Starting Optuna Trial {trial.number} (Maximize NSE with copied JIT step) ---")
        print(f"Parameters: {trial.params}")
        # Ensure correct DTYPE is set based on config before running trial
        global DTYPE
        DTYPE = jnp.dtype(config["device"]["dtype"])

        # Call the function from the modified hyperopt/hpo_train.py
        trial_nse = run_hpo_trial_with_nse(config)

        # Check for NaN/Inf return values from the trial function
        if not isinstance(trial_nse, (int, float)) or not np.isfinite(trial_nse):
             print(f"Trial {trial.number} returned invalid NSE ({trial_nse}). Reporting -inf.")
             trial_nse = -float('inf')

    except optuna.TrialPruned as e:
        print(f"Trial {trial.number} pruned: {e}")
        raise # Re-raise prune exception for Optuna
    except Exception as e:
        print(f"Trial {trial.number} failed with an error: {e}")
        import traceback
        traceback.print_exc()
        trial_nse = -float('inf') # Report failure to Optuna

    finally:
        print(f"--- Finished Optuna Trial {trial.number} with NSE: {trial_nse:.6f} ---")
        # No artifact cleanup needed here as hpo_train doesn't save models/plots

    # 4. Return the final NSE (Optuna aims to maximize this)
    return trial_nse


# --- Function to Save Top Trial Configs (same as before) ---
def save_best_trials(study: optuna.Study, base_config_path: str, save_dir: str, num_trials: int = 3):
    """Saves the configurations of the top N trials based on NSE."""
    print(f"\n--- Saving Top {num_trials} Trial Configurations ---")
    try:
        # Get completed trials sorted by value (NSE) in descending order
        best_trials = sorted(
            study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]),
            key=lambda t: t.value if t.value is not None and np.isfinite(t.value) else -float('inf'), # Handle None/NaN/Inf
            reverse=True # Maximize NSE
        )

        valid_best_trials = [t for t in best_trials if t.value is not None and np.isfinite(t.value) and t.value > -float('inf')]

        if not valid_best_trials:
            print("No trials completed successfully with valid NSE. Nothing to save.")
            return

        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)

        for i, trial in enumerate(valid_best_trials[:num_trials]):
            print(f"\nSaving config for Rank {i+1} (Trial {trial.number}, NSE: {trial.value:.6f})")
            try:
                # Reload base config
                best_config = load_config(base_config_path)
                best_config = dict(best_config)

                # Apply trial parameters - More robust application logic
                for key, value in trial.params.items():
                    levels = key.split('_')
                    d = best_config
                    try:
                        # Navigate through nested dictionaries
                        for level in levels[:-1]:
                            if level.startswith('ic') or level.startswith('bc'):
                                # Handle combined ic_bc_grid key structure
                                if 'ic_bc_grid' not in d: d['ic_bc_grid'] = {}
                                d = d['ic_bc_grid']
                                break # Assume the rest is the key within ic_bc_grid
                            elif level.startswith('building') and len(levels)>1:
                                if 'building' not in d: d['building'] = {}
                                d = d['building']
                                break
                            elif level.startswith('loss'):
                                if 'loss_weights' not in d: d['loss_weights'] = {}
                                d = d['loss_weights']
                                break
                            elif level in d and isinstance(d[level], dict):
                                d = d[level]
                            else: # If intermediate level not found or not dict, stop
                                print(f"Warning: Could not fully resolve path for key '{key}' at level '{level}'. Assigning higher up.")
                                break

                        # Determine final key name
                        final_key = '_'.join(levels[levels.index(level):]) if level in levels[:-1] else levels[-1]
                        # Special handling for loss weights format
                        if 'loss_weights' in d and not final_key.endswith('_weight'):
                            final_key += '_weight'

                        # Assign value
                        if isinstance(d, dict):
                            d[final_key] = value
                        else: # If d is not a dict at the end, assign to the original dict
                            print(f"Warning: Target for key '{key}' is not a dict. Assigning to base level.")
                            best_config[key] = value

                    except (KeyError, IndexError, TypeError) as e:
                        print(f"  Warning: Could not set parameter '{key}': {e}. Assigning top-level.")
                        best_config[key] = value # Fallback


                # Clean up specific keys
                best_config.get('loss_weights', {}).pop('data_weight', None) # Ensure data_weight is removed if it reappeared
                best_config.pop('CONFIG_PATH', None)

                # Restore original training duration settings from base config
                base_cfg_full = load_config(base_config_path)
                best_config['training']['epochs'] = base_cfg_full['training'].get('epochs', 20000) # Use original or default
                best_config['device']['early_stop_min_epochs'] = base_cfg_full['device'].get('early_stop_min_epochs', 4000)
                best_config['device']['early_stop_patience'] = base_cfg_full['device'].get('early_stop_patience', 3000)

                # Save the reconstructed config
                save_path = os.path.join(save_dir, f"{study.study_name}_rank_{i+1}_trial_{trial.number}.yaml")
                with open(save_path, 'w') as f:
                    yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
                print(f"  Config saved to: {save_path}")

            except Exception as e_save:
                print(f"  Error reconstructing or saving config for trial {trial.number}: {e_save}")
                import traceback
                traceback.print_exc()

    except Exception as e_fetch:
        print(f"Error fetching or processing best trials: {e_fetch}")


# --- Main Execution Block ---
if __name__ == "__main__":
    os.makedirs(os.path.dirname(STORAGE_PATH), exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=42) # Using TPESampler
    # Pruning based on intermediate NSE is tricky, using NopPruner
    pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        load_if_exists=True,
        direction="maximize", # Maximize NSE
        sampler=sampler,
        pruner=pruner
    )

    print(f"Starting/Resuming NSE optimization study '{STUDY_NAME}'...")
    print(f"Database located at: {STORAGE_PATH}")
    print(f"Number of trials requested: {N_TRIALS}")
    print(f"Current number of trials in study: {len(study.trials)}")

    try:
        study.optimize(
            objective,
            n_trials=N_TRIALS,
            gc_after_trial=True # Enable garbage collection between trials
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during study.optimize: {e}")
        import traceback
        traceback.print_exc()

    # --- Print Summary and Save Top 3 Valid Trials ---
    print("\n--- Optimization Finished ---")
    try:
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        failed_trials = study.get_trials(deepcopy=False, states=[TrialState.FAIL])

        print(f"Study statistics: ")
        print(f"  Finished trials: {len(study.trials)}")
        print(f"  Complete trials: {len(complete_trials)}")
        print(f"  Pruned trials: {len(pruned_trials)}")
        print(f"  Failed trials: {len(failed_trials)}")

        # Filter out failed trials BEFORE finding the best
        valid_complete_trials = [t for t in complete_trials if t.value is not None and np.isfinite(t.value) and t.value > -float('inf')]

        if valid_complete_trials:
             # Sort valid trials to find the best
             best_valid_trial = max(valid_complete_trials, key=lambda t: t.value)
             print("\nBest valid trial:")
             print(f"  Trial Number: {best_valid_trial.number}")
             print(f"  Value (Best NSE): {best_valid_trial.value:.6f}")
             print("  Params: ")
             for key, value in best_valid_trial.params.items(): print(f"    {key}: {value}")

             # Save top 3 valid trials
             save_best_trials(study, BASE_CONFIG_PATH, save_dir=os.path.dirname(STORAGE_PATH), num_trials=3)

        else:
            print("\nNo trials completed successfully with a valid NSE.")
            # Optionally, print info about failed trials if needed
            if failed_trials:
                 print(f"\n{len(failed_trials)} trials failed.")
            elif complete_trials:
                 print("\nAll completed trials resulted in invalid NSE values.")


    except Exception as e:
        print(f"Error retrieving or saving study results: {e}")