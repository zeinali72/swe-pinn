# hyperopt/hyper_optimizer.py
import os
import yaml
import optuna
from optuna.trial import TrialState
import shutil
import time

# --- Use train_test.py as the main training entry point ---
from src.train_test import main as train_main # Make sure src/train_test.py has the run_for_hpo flag

# --- Configuration ---
BASE_CONFIG_PATH = "experiments/one_building_config.yaml"
N_TRIALS = 50  # Number of optimization trials to run
STUDY_NAME = "swe-pinn-optimization-resolutions-noaim" # New study name
STORAGE_PATH = "sqlite:///hyperopt/swe_pinn_opt_resolutions_noaim.db" # New DB path

# --- HPO Specific Training Settings ---
HPO_EPOCHS = 5000
HPO_EARLY_STOP_MIN_EPOCHS = 1000
HPO_EARLY_STOP_PATIENCE = 1500

# --- Objective Function for Optuna ---
def objective(trial: optuna.trial.Trial) -> float:
    """
    Defines a single optimization trial.
    Optuna will call this function multiple times with different hyperparameter suggestions.
    """
    # 1. Load the base configuration
    try:
        with open(BASE_CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Base config file not found at {BASE_CONFIG_PATH}")
        return -1.0 # Indicate failure

    # --- 2. Suggest Hyperparameters ---
    config['training']['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    config['training']['batch_size'] = trial.suggest_categorical("batch_size", [256, 512, 1024])
    config['model']['width'] = trial.suggest_categorical("model_width", [128, 256, 512])
    config['model']['depth'] = trial.suggest_int("model_depth", 3, 7)
    config['model']['ff_dims'] = trial.suggest_categorical("ff_dims", [256, 512, 1024])
    config['model']['fourier_scale'] = trial.suggest_float("fourier_scale", 5.0, 25.0)
    config['loss_weights']['pde_weight'] = trial.suggest_float("pde_weight", 1e2, 1e7, log=True)
    config['loss_weights']['bc_weight'] = trial.suggest_float("bc_weight", 1e0, 1e4, log=True)
    config['loss_weights']['ic_weight'] = trial.suggest_float("ic_weight", 1e0, 1e4, log=True)
    if 'building' in config:
        config['loss_weights']['building_bc_weight'] = trial.suggest_float("building_bc_weight", 1e1, 1e5, log=True)

    config['grid']['nx'] = trial.suggest_int("grid_nx", 30, 100)
    config['grid']['ny'] = trial.suggest_int("grid_ny", 10, 50)
    config['grid']['nt'] = trial.suggest_int("grid_nt", 10, 50)
    ic_scale = trial.suggest_int("ic_bc_scale_xy", 20, 80)
    bc_scale_y = trial.suggest_int("ic_bc_scale_y", 15, 50)
    bc_scale_t = trial.suggest_int("ic_bc_scale_t", 10, 40)
    config['ic_bc_grid']['nx_ic'] = ic_scale
    config['ic_bc_grid']['ny_ic'] = ic_scale
    config['ic_bc_grid']['ny_bc_left'] = bc_scale_y
    config['ic_bc_grid']['nt_bc_left'] = bc_scale_t
    config['ic_bc_grid']['ny_bc_right'] = bc_scale_y
    config['ic_bc_grid']['nt_bc_right'] = bc_scale_t
    config['ic_bc_grid']['nx_bc_bottom'] = ic_scale
    config['ic_bc_grid']['nt_bc_other'] = bc_scale_t
    config['ic_bc_grid']['nx_bc_top'] = ic_scale
    if 'building' in config:
        bldg_scale_xy = trial.suggest_int("bldg_scale_xy", 10, 40)
        bldg_scale_t = trial.suggest_int("bldg_scale_t", 10, 40)
        config['building']['nx'] = bldg_scale_xy
        config['building']['ny'] = bldg_scale_xy
        config['building']['nt'] = bldg_scale_t

    config['training']['epochs'] = HPO_EPOCHS
    config['device']['early_stop_min_epochs'] = HPO_EARLY_STOP_MIN_EPOCHS
    config['device']['early_stop_patience'] = HPO_EARLY_STOP_PATIENCE

    # 3. Create a temporary config file for this specific trial
    trial_config_dir = os.path.join("hyperopt", "trial_configs")
    os.makedirs(trial_config_dir, exist_ok=True)
    trial_config_path = os.path.join(trial_config_dir, f"config_trial_{trial.number}.yaml")

    with open(trial_config_path, 'w') as f:
        yaml.dump(config, f)

    best_nse = -1.0
    trial_successful = False
    try:
        # 4. Run the training script with the HPO flag
        print(f"\n--- Starting Trial {trial.number} ---")
        print(f"Config: {trial_config_path}")
        print(f"Parameters: {trial.params}")

        # Pass run_for_hpo=True to skip saving prompt and plots
        best_nse = train_main(trial_config_path, run_for_hpo=True)
        if best_nse != best_nse or abs(best_nse) == float('inf'): # Check for NaN/Inf
             print(f"Trial {trial.number} resulted in invalid NSE ({best_nse}). Returning -1.0.")
             best_nse = -1.0
        else:
            trial_successful = True

    except Exception as e:
        print(f"Trial {trial.number} failed during training with an error: {e}")
        import traceback
        traceback.print_exc()
        best_nse = -1.0

    finally:
        # --- Clean up trial artifacts ---
        print(f"--- Cleaning up artifacts for Trial {trial.number} ---")
        config_basename = os.path.splitext(os.path.basename(trial_config_path))[0]

        def find_and_remove_dirs(base_dir, prefix):
             if not os.path.exists(base_dir): return
             try:
                 for item in os.listdir(base_dir):
                     item_path = os.path.join(base_dir, item)
                     if os.path.isdir(item_path) and prefix in item:
                         try:
                             shutil.rmtree(item_path)
                             print(f"Cleaned up {item_path}")
                         except OSError as e:
                             print(f"Error removing {item_path}: {e}. Retrying...")
                             time.sleep(1)
                             try: shutil.rmtree(item_path); print(f"Cleaned up {item_path} after retry.")
                             except OSError as e2: print(f"Failed to remove {item_path} even after retry: {e2}")
             except Exception as e: print(f"Error during cleanup in {base_dir}: {e}")

        find_and_remove_dirs("results", config_basename)
        find_and_remove_dirs("models", config_basename)
        # No Aim cleanup needed

        try:
            if os.path.exists(trial_config_path):
                os.remove(trial_config_path)
        except OSError as e:
            print(f"Error removing temporary config {trial_config_path}: {e}")

        return float(best_nse)


# --- Main Execution Block ---
if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=42)
    # Pruning requires trial.report in train_main - disable if not implemented
    # pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=int(HPO_EPOCHS * 0.4), interval_steps=int(HPO_EPOCHS*0.1))
    pruner = optuna.pruners.NopPruner() # No pruning

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )

    print(f"Starting/Resuming optimization study '{STUDY_NAME}'...")
    print(f"Database located at: {STORAGE_PATH}")
    print(f"Number of trials requested: {N_TRIALS}")
    print(f"Current number of trials in study: {len(study.trials)}")

    # --- REMOVED AimCallback ---

    try:
        # Pass callbacks=None since we removed AimCallback
        study.optimize(objective, n_trials=N_TRIALS, callbacks=None, gc_after_trial=True)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during study.optimize: {e}")
        import traceback
        traceback.print_exc()

    # --- Print Results ---
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

        if complete_trials:
             print("\nBest trial:")
             trial = study.best_trial
             print(f"  Trial Number: {trial.number}")
             print(f"  Value (Best NSE): {trial.value:.6f}")
             print("  Params: ")
             for key, value in trial.params.items(): print(f"    {key}: {value}")
        else:
            print("\nNo trials completed successfully.")
    except Exception as e:
        print(f"Error retrieving study results: {e}")