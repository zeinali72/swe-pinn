# /workspaces/swe-pinn/hyperopt/hpo_optimizer.py
import os
import yaml
import optuna
from optuna.trial import TrialState
import time
import sys
import shutil
import numpy as np # For checking finite values
import jax.numpy as jnp # <<<--- ADDED THIS IMPORT

# --- Ensure src and hyperopt modules can be imported ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hyperopt.hpo_train import run_hpo_trial_with_nse # Import NSE trial runner
from src.config import load_config, DTYPE # Load base config, set DTYPE

# --- Configuration ---
BASE_CONFIG_PATH = "experiments/one_building_config.yaml"
N_TRIALS = 50
STUDY_NAME = "swe-pinn-nse-hpo-building-v2" # Updated study name
STORAGE_PATH = f"sqlite:///hyperopt/{STUDY_NAME}.db" # Updated DB path

# --- HPO Specific Training Settings ---
HPO_EPOCHS = 5000
HPO_EARLY_STOP_MIN_EPOCHS = 1000
HPO_EARLY_STOP_PATIENCE = 1500

# --- Objective Function for Optuna ---
def objective(trial: optuna.trial.Trial) -> float:
    """
    Defines a single optimization trial targeting NSE maximization.
    Suggests grid parameters directly.
    """
    # 1. Load the base configuration
    try:
        # Load config also sets global DTYPE from src.config
        config = load_config(BASE_CONFIG_PATH)
        # Convert back to regular dict for modification
        config = dict(config)
    except FileNotFoundError:
        print(f"ERROR: Base config file not found at {BASE_CONFIG_PATH}")
        # Return a value indicating failure that Optuna understands for maximization (-inf)
        return -float('inf')

    # --- 2. Suggest Hyperparameters ---
    # Training
    config['training']['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    config['training']['batch_size'] = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])

    # Model (assuming FourierPINN)
    if config['model']['name'] == "FourierPINN":
        config['model']['width'] = trial.suggest_categorical("model_width", [128, 256, 512])
        config['model']['depth'] = trial.suggest_int("model_depth", 3, 7)
        config['model']['ff_dims'] = trial.suggest_categorical("ff_dims", [256, 512, 1024])
        config['model']['fourier_scale'] = trial.suggest_float("fourier_scale", 5.0, 25.0)

    # Loss weights (Excluding data_weight)
    config['loss_weights']['pde_weight'] = trial.suggest_float("pde_weight", 1e2, 1e7, log=True)
    config['loss_weights']['bc_weight'] = trial.suggest_float("bc_weight", 1e0, 1e4, log=True)
    config['loss_weights']['ic_weight'] = trial.suggest_float("ic_weight", 1e0, 1e4, log=True)
    config['loss_weights']['building_bc_weight'] = trial.suggest_float("building_bc_weight", 1e1, 1e5, log=True)
    if 'data_weight' in config['loss_weights']: del config['loss_weights']['data_weight']

    # --- Grid Resolution params (Suggest directly) ---
    config['grid']['nx'] = trial.suggest_int("grid_nx", 30, 100)
    config['grid']['ny'] = trial.suggest_int("grid_ny", 10, 50)
    config['grid']['nt'] = trial.suggest_int("grid_nt", 10, 50)

    # IC/BC Grid Points
    config['ic_bc_grid']['nx_ic'] = trial.suggest_int("nx_ic", 20, 80)
    config['ic_bc_grid']['ny_ic'] = trial.suggest_int("ny_ic", 20, 80)
    config['ic_bc_grid']['ny_bc_left'] = trial.suggest_int("ny_bc_left", 15, 50)
    config['ic_bc_grid']['nt_bc_left'] = trial.suggest_int("nt_bc_left", 10, 40)
    config['ic_bc_grid']['ny_bc_right'] = trial.suggest_int("ny_bc_right", 15, 50)
    config['ic_bc_grid']['nt_bc_right'] = trial.suggest_int("nt_bc_right", 10, 40)
    config['ic_bc_grid']['nx_bc_bottom'] = trial.suggest_int("nx_bc_bottom", 20, 80)
    config['ic_bc_grid']['nt_bc_other'] = trial.suggest_int("nt_bc_other", 10, 40) # Used for bottom and top time points
    config['ic_bc_grid']['nx_bc_top'] = trial.suggest_int("nx_bc_top", 20, 80)

    # Building Grid Points
    config['building']['nx'] = trial.suggest_int("building_nx", 10, 40) # Bottom/Top walls X points
    config['building']['ny'] = trial.suggest_int("building_ny", 10, 40) # Left/Right walls Y points
    config['building']['nt'] = trial.suggest_int("building_nt", 10, 40) # Time points for building walls

    # HPO training settings
    config['training']['epochs'] = HPO_EPOCHS
    config['device']['early_stop_min_epochs'] = HPO_EARLY_STOP_MIN_EPOCHS
    config['device']['early_stop_patience'] = HPO_EARLY_STOP_PATIENCE

    # --- 3. Run the training trial with NSE evaluation ---
    trial_nse = -float('inf')
    try:
        print(f"\n--- Starting Optuna Trial {trial.number} (Maximize NSE) ---")
        # Ensure DTYPE is set correctly before calling the trial runner
        # Need to use jnp here, hence the added import
        global DTYPE
        DTYPE = jnp.dtype(config["device"]["dtype"]) # Use jnp here
        trial_nse = run_hpo_trial_with_nse(config)

        # Check for NaN/Inf return values
        if not isinstance(trial_nse, (int, float)) or not np.isfinite(trial_nse):
             print(f"Trial {trial.number} returned invalid NSE ({trial_nse}). Reporting -inf.")
             trial_nse = -float('inf')

    except optuna.TrialPruned as e:
        print(f"Trial {trial.number} pruned: {e}")
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed with an error: {e}")
        import traceback
        traceback.print_exc()
        trial_nse = -float('inf')

    finally:
        print(f"--- Finished Optuna Trial {trial.number} with NSE: {trial_nse:.6f} ---")

    # 4. Return the final NSE (Optuna aims to maximize this)
    return trial_nse


def save_best_trials(study: optuna.Study, base_config_path: str, save_dir: str, num_trials: int = 3):
    """Saves the configurations of the top N trials."""
    print(f"\n--- Saving Top {num_trials} Trial Configurations ---")
    try:
        # Get completed trials sorted by value (NSE) in descending order
        best_trials = sorted(
            study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]),
            key=lambda t: t.value if t.value is not None else -float('inf'), # Handle None values
            reverse=True # Maximize NSE
        )

        if not best_trials:
            print("No trials completed successfully. Nothing to save.")
            return

        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)

        for i, trial in enumerate(best_trials[:num_trials]):
             # Skip trials that failed and reported -inf
            if trial.value == -float('inf'):
                 print(f"\nSkipping Rank {i+1} (Trial {trial.number}) due to failure (NSE: -inf)")
                 continue

            print(f"\nSaving config for Rank {i+1} (Trial {trial.number}, NSE: {trial.value:.6f})")
            try:
                # Reload base config to avoid carrying over modifications
                best_config = load_config(base_config_path)
                best_config = dict(best_config)

                # Apply trial parameters (simplified logic, assumes keys match)
                for key, value in trial.params.items():
                    parts = key.split('_')
                    target_dict = best_config
                    category = parts[0]
                    if category in ['ic', 'bc']: category = 'ic_bc_grid'
                    if category == 'loss': category = 'loss_weights'

                    if category in best_config and isinstance(best_config[category], dict):
                        sub_key_parts = parts[1:] if category not in ['learning', 'batch', 'fourier', 'pde', 'ic', 'bc', 'building'] else parts # Handle single-word keys
                        # Special handling for potentially nested keys like ic_bc_grid
                        if category == 'ic_bc_grid' and len(parts) > 2:
                            sub_key = '_'.join(parts[1:]) # e.g. nx_ic
                        elif category == 'building' and len(parts) > 1:
                            sub_key = '_'.join(parts[1:]) # e.g. building_nx
                        elif category in ['model', 'training', 'loss_weights', 'grid', 'device'] and len(parts) > 1:
                             sub_key = '_'.join(parts[1:])
                        elif len(parts) == 1: # like 'learning_rate', 'batch_size'
                            sub_key = key
                        else: # Fallback or simple keys like grid_nx
                            sub_key = '_'.join(parts[1:]) if len(parts) > 1 else parts[0]


                        # Refined check for sub_key existence
                        if sub_key and sub_key in best_config[category]:
                             best_config[category][sub_key] = value
                        elif sub_key: # Assign even if sub_key is new within category
                             best_config[category][sub_key] = value
                        else: # Fallback for keys like learning_rate assigned directly in training
                            best_config[category][key] = value # Less robust, might misplace keys
                    else:
                        best_config[key] = value # Fallback: assign top-level

                # Re-ensure specific keys are correct or removed
                if 'data_weight' in best_config.get('loss_weights', {}):
                    del best_config['loss_weights']['data_weight']
                if 'CONFIG_PATH' in best_config:
                    del best_config['CONFIG_PATH']

                # Restore original training duration settings
                base_cfg_train = load_config(base_config_path)['training']
                base_cfg_dev = load_config(base_config_path)['device']
                best_config['training']['epochs'] = base_cfg_train.get('epochs') # Use original full epochs
                best_config['device']['early_stop_min_epochs'] = base_cfg_dev.get('early_stop_min_epochs')
                best_config['device']['early_stop_patience'] = base_cfg_dev.get('early_stop_patience')


                # Save the reconstructed config
                save_path = os.path.join(save_dir, f"{study.study_name}_rank_{i+1}_trial_{trial.number}.yaml")
                with open(save_path, 'w') as f:
                    yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
                print(f"  Config saved to: {save_path}")

            except Exception as e_save:
                print(f"  Error reconstructing or saving config for trial {trial.number}: {e_save}")
                import traceback
                traceback.print_exc() # Print detailed traceback for saving issues

    except Exception as e_fetch:
        print(f"Error fetching or processing best trials: {e_fetch}")


# --- Main Execution Block ---
if __name__ == "__main__":
    os.makedirs(os.path.dirname(STORAGE_PATH), exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.NopPruner() # No pruning for now

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
            gc_after_trial=True
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during study.optimize: {e}")
        import traceback
        traceback.print_exc()

    # --- Print Summary and Save Top 3 ---
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

        # Filter out failed trials before finding the best
        valid_complete_trials = [t for t in complete_trials if t.value is not None and np.isfinite(t.value) and t.value != -float('inf')]


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
    except Exception as e:
        print(f"Error retrieving or saving study results: {e}")