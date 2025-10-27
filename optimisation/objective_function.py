# optimisation/objective_function.py
"""
Defines the Optuna objective function: suggests hyperparameters,
builds the trial configuration, and calls the training loop.
"""
import optuna
from flax.core import FrozenDict
import jax.numpy as jnp # Only needed for type hints potentially
from typing import Dict, Any
import copy

# Import the training loop function from the same directory
from optimization_train_loop import run_training_trial

def objective(trial: optuna.trial.Trial, base_config_dict: Dict, data_free: bool) -> float:
    """
    Objective function for Optuna hyperparameter optimization.
    Suggests hyperparameters, constructs the config, runs training, returns NSE.
    """
    base_cfg = FrozenDict(base_config_dict)
    model_name = base_cfg["model"]["name"]
    has_building = "building" in base_cfg
    # GradNorm enabled only if specified in base config AND not data_free
    enable_gradnorm = base_cfg.get("gradnorm", {}).get("enable", False) and not data_free

    # --- 1. Define Hyperparameter Search Space ---
    # print(f"Trial {trial.number}: Suggesting hyperparameters...") # Keep logging minimal
    trial_params = {}

    # === Training Hyperparameters ===
    trial_params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    trial_params["batch_size"] = trial.suggest_categorical("batch_size", [256, 512, 1024])
    # Epochs are fixed via run_optimization.py --opt_epochs argument

    # === Model Hyperparameters ===
    trial_params["model_width"] = trial.suggest_categorical("model_width", [64, 128, 256, 512])
    trial_params["model_depth"] = trial.suggest_int("model_depth", 2, 6)

    if model_name == "FourierPINN":
        trial_params["ff_dims"] = trial.suggest_categorical("ff_dims", [64, 128, 256, 512])
        trial_params["fourier_scale"] = trial.suggest_float("fourier_scale", 1.0, 20.0)
    elif model_name == "SIREN":
        trial_params["w0"] = trial.suggest_float("w0", 1.0, 30.0)
    # Add parameters for DGMNetwork if needed

    # === Grid Hyperparameters ===
    # Base grid for PDE
    nx_base = trial.suggest_int("nx_base", 20, 80, step=4)
    ny_base = trial.suggest_int("ny_base", 10, 40, step=2)
    nt_base = trial.suggest_int("nt_base", 10, 40, step=2)
    trial_params["grid"] = {"nx": nx_base, "ny": ny_base, "nt": nt_base}

    # Factors for IC/BC grids
    trial_params["ic_bc_grid"] = {
        "nx_ic": max(1, int(trial.suggest_float("nx_ic_factor", 0.5, 1.5) * nx_base)),
        "ny_ic": max(1, int(trial.suggest_float("ny_ic_factor", 0.5, 1.5) * ny_base)),
        "ny_bc_left": max(1, int(trial.suggest_float("ny_bc_left_factor", 0.3, 1.2) * ny_base)),
        "nt_bc_left": max(1, int(trial.suggest_float("nt_bc_left_factor", 0.3, 1.2) * nt_base)),
        "ny_bc_right": max(1, int(trial.suggest_float("ny_bc_right_factor", 0.3, 1.2) * ny_base)),
        "nt_bc_right": max(1, int(trial.suggest_float("nt_bc_right_factor", 0.3, 1.2) * nt_base)),
        "nx_bc_bottom": max(1, int(trial.suggest_float("nx_bc_bottom_factor", 0.5, 1.5) * nx_base)),
        "nt_bc_other": max(1, int(trial.suggest_float("nt_bc_other_factor", 0.3, 1.2) * nt_base)),
        "nx_bc_top": max(1, int(trial.suggest_float("nx_bc_top_factor", 0.5, 1.5) * nx_base)),
    }

    # Factors for Building grid (if applicable)
    if has_building:
         trial_params["building_grid"] = {
             "nx": max(1, int(trial.suggest_float("nx_bldg_factor", 0.3, 1.2) * nx_base)),
             "ny": max(1, int(trial.suggest_float("ny_bldg_factor", 0.3, 1.2) * ny_base)),
             "nt": max(1, int(trial.suggest_float("nt_bldg_factor", 0.3, 1.2) * nt_base)),
         }

    # === Loss Weights / GradNorm Hyperparameters ===
    trial_params["loss_weights"] = {}
    if enable_gradnorm: # Optimize GradNorm alpha (only if not data_free)
        trial_params["gradnorm_alpha"] = trial.suggest_float("gradnorm_alpha", 0.1, 3.0) # Wider range maybe
        # Set initial weights to 1.0; GradNorm adjusts them
        trial_params["loss_weights"]["pde_weight"] = 1.0
        trial_params["loss_weights"]["ic_weight"] = 1.0
        trial_params["loss_weights"]["bc_weight"] = 1.0
        if has_building: trial_params["loss_weights"]["building_bc_weight"] = 1.0
        # Data weight is included because enable_gradnorm is only true if not data_free
        trial_params["loss_weights"]["data_weight"] = 1.0
    else: # Optimize static weights
        # Fix PDE weight to 1.0 if data_free, otherwise suggest its scale
        if data_free:
            trial_params["loss_weights"]["pde_weight"] = 1.0
        else:
            trial_params["loss_weights"]["pde_weight"] = trial.suggest_float("pde_weight_scale", 1e-1, 1e4, log=True) # Wider range

        # Suggest factors relative to the (potentially suggested) PDE weight
        trial_params["loss_weights"]["ic_weight"] = trial_params["loss_weights"]["pde_weight"] * trial.suggest_float("ic_weight_factor", 1e-3, 1e3, log=True)
        trial_params["loss_weights"]["bc_weight"] = trial_params["loss_weights"]["pde_weight"] * trial.suggest_float("bc_weight_factor", 1e-3, 1e3, log=True)
        if has_building:
             trial_params["loss_weights"]["building_bc_weight"] = trial_params["loss_weights"]["pde_weight"] * trial.suggest_float("building_bc_weight_factor", 1e-3, 1e3, log=True)
        # Data weight factor only optimized if not data_free
        if not data_free:
             trial_params["loss_weights"]["data_weight"] = trial_params["loss_weights"]["pde_weight"] * trial.suggest_float("data_weight_factor", 1e-3, 1e3, log=True)
        else:
             trial_params["loss_weights"]["data_weight"] = 0.0 # Explicitly zero

    # === Construct Trial Configuration Dictionary ===
    trial_config_dict = copy.deepcopy(base_config_dict) # Deep copy is crucial

    # --- Update sections with suggested hyperparameters ---
    trial_config_dict["training"]["learning_rate"] = trial_params["learning_rate"]
    trial_config_dict["training"]["batch_size"] = trial_params["batch_size"]
    trial_config_dict["training"]["epochs"] = base_cfg["training"]["opt_epochs"] # Use fixed opt_epochs

    trial_config_dict["model"]["width"] = trial_params["model_width"]
    trial_config_dict["model"]["depth"] = trial_params["model_depth"]
    if model_name == "FourierPINN":
        trial_config_dict["model"]["ff_dims"] = trial_params["ff_dims"]
        trial_config_dict["model"]["fourier_scale"] = trial_params["fourier_scale"]
    elif model_name == "SIREN":
        trial_config_dict["model"]["w0"] = trial_params["w0"]

    trial_config_dict["grid"] = trial_params["grid"]
    trial_config_dict["ic_bc_grid"] = trial_params["ic_bc_grid"]
    if has_building:
         # Make sure 'building' key exists before updating sub-keys
         if "building" not in trial_config_dict: trial_config_dict["building"] = {}
         trial_config_dict["building"]["nx"] = trial_params["building_grid"]["nx"]
         trial_config_dict["building"]["ny"] = trial_params["building_grid"]["ny"]
         trial_config_dict["building"]["nt"] = trial_params["building_grid"]["nt"]

    trial_config_dict["loss_weights"] = trial_params["loss_weights"]
    if "gradnorm" not in trial_config_dict: trial_config_dict["gradnorm"] = {} # Ensure key exists
    if enable_gradnorm:
        trial_config_dict["gradnorm"]["enable"] = True
        trial_config_dict["gradnorm"]["alpha"] = trial_params["gradnorm_alpha"]
        # Keep LR/freq from base or optimize them too
        trial_config_dict["gradnorm"]["learning_rate"] = base_cfg.get("gradnorm",{}).get("learning_rate", 0.01)
        trial_config_dict["gradnorm"]["update_freq"] = base_cfg.get("gradnorm",{}).get("update_freq", 100)
    else:
        trial_config_dict["gradnorm"]["enable"] = False


    # Convert final config to FrozenDict for JAX functions
    trial_cfg_frozen = FrozenDict(trial_config_dict)

    # --- Run the training trial ---
    try:
        # Pass the Optuna trial object itself for pruning callbacks
        best_nse = run_training_trial(trial, trial_cfg_frozen, data_free)

        # Ensure a standard float is returned for Optuna
        if isinstance(best_nse, (jax.Array, np.ndarray)):
            best_nse = float(best_nse)
        # Handle cases where training failed or NSE was invalid
        if best_nse <= -jnp.inf or jnp.isnan(best_nse):
             print(f"Trial {trial.number}: Invalid NSE ({best_nse}). Returning -1.0.")
             return -1.0

        return best_nse

    except optuna.exceptions.TrialPruned as e:
         # print(f"Trial {trial.number} pruned from within objective.") # Optional logging
         raise e # Re-raise for Optuna to handle pruning correctly
    except Exception as e:
         print(f"Trial {trial.number}: UNHANDLED EXCEPTION in run_training_trial: {e}")
         import traceback
         traceback.print_exc()
         # Return a very poor score to Optuna to indicate failure
         return -1.0