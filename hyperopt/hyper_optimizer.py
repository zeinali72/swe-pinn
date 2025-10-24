import os
import yaml
import optuna
from optuna.trial import TrialState
import shutil

# --- Important: This assumes your train_test.py is now merged or is the main training file ---
# If you have merged train_test.py into train.py, change the import below
from src.train_test import main as train_main

# --- Configuration ---
BASE_CONFIG_PATH = "experiments/one_building_config.yaml"
N_TRIALS = 50  # Number of optimization trials to run
STUDY_NAME = "swe-pinn-optimization"
STORAGE_PATH = "sqlite:///hyperopt/swe_pinn_opt.db"

# --- Objective Function for Optuna ---
def objective(trial: optuna.trial.Trial) -> float:
    """
    Defines a single optimization trial.
    Optuna will call this function multiple times with different hyperparameter suggestions.
    """
    # 1. Load the base configuration
    with open(BASE_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # --- 2. Suggest Hyperparameters ---
    # We use suggest_float with log=True for parameters that can span several orders of magnitude.
    
    # Training parameters
    config['training']['learning_rate'] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Model architecture
    config['model']['width'] = trial.suggest_categorical("model_width", [128, 256, 512])
    config['model']['depth'] = trial.suggest_int("model_depth", 3, 6)
    config['model']['ff_dims'] = trial.suggest_categorical("ff_dims", [256, 512])
    config['model']['fourier_scale'] = trial.suggest_float("fourier_scale", 5.0, 20.0)

    # Loss weights (the most critical part for PINNs)
    config['loss_weights']['pde_weight'] = trial.suggest_float("pde_weight", 1e2, 1e7, log=True)
    config['loss_weights']['bc_weight'] = trial.suggest_float("bc_weight", 1e0, 1e3, log=True)
    config['loss_weights']['ic_weight'] = trial.suggest_float("ic_weight", 1e0, 1e3, log=True)
    config['loss_weights']['building_bc_weight'] = trial.suggest_float("building_bc_weight", 1e1, 1e4, log=True)

    # For faster optimization, you might want to reduce epochs during the search
    # config['training']['epochs'] = 10000 
    
    # 3. Create a temporary config file for this specific trial
    trial_config_dir = os.path.join("hyperopt", "trial_configs")
    os.makedirs(trial_config_dir, exist_ok=True)
    trial_config_path = os.path.join(trial_config_dir, f"config_trial_{trial.number}.yaml")
    
    with open(trial_config_path, 'w') as f:
        yaml.dump(config, f)

    try:
        # 4. Run the training script and get the score (best NSE)
        print(f"\n--- Starting Trial {trial.number} ---")
        best_nse = train_main(trial_config_path)
        
        # --- Clean up trial artifacts to save space ---
        # The main results are stored in the Optuna database.
        trial_name_prefix = f"config_trial_{trial.number}"
        for dir_name in ["results", "models", "aim_repo/runs"]:
            if os.path.exists(dir_name):
                for item in os.listdir(dir_name):
                    if trial_name_prefix in item:
                        item_path = os.path.join(dir_name, item)
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                            print(f"Cleaned up {item_path}")

        return best_nse

    except Exception as e:
        print(f"Trial {trial.number} failed with an error: {e}")
        # Tell Optuna this trial failed
        return -1.0 # Return a very bad score


# --- Main Execution Block ---
if __name__ == "__main__":
    # Optuna uses a sampler to decide which parameters to try next. TPE is a good Bayesian default.
    sampler = optuna.samplers.TPESampler(seed=42)
    
    # A pruner can stop unpromising trials early.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3000, interval_steps=1000)

    # Create or load the study. This allows you to resume the optimization later.
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_PATH,
        load_if_exists=True,
        direction="maximize",  # We want to maximize the NSE
        sampler=sampler,
        pruner=pruner
    )

    print(f"Starting optimization study '{STUDY_NAME}'...")
    print(f"Database located at: {STORAGE_PATH}")
    
    # Start the optimization
    study.optimize(objective, n_trials=N_TRIALS)

    # --- Print Results ---
    print("\n--- Optimization Finished ---")
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print(f"Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")

    print("\nBest trial:")
    trial = study.best_trial

    print(f"  Value (Best NSE): {trial.value:.6f}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")