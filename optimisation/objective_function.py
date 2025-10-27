# optimisation/objective_function.py
"""
Defines the Optuna objective function: suggests hyperparameters,
builds the trial configuration, and calls the training loop.
"""
import jax
import optuna
from flax.core import FrozenDict
import jax.numpy as jnp # Only needed for type hints potentially
from typing import Dict, Any
import copy

# Import the training loop function from the same directory
from optimization_train_loop import run_training_trial

def objective(trial: optuna.trial.Trial,
              base_config_dict: Dict,
              data_free: bool,
              enable_gradnorm: bool) -> float: # Accept both flags
    """
    Objective function for Optuna hyperparameter optimization.
    Suggests hyperparameters based on data_free and enable_gradnorm flags,
    constructs the config, runs training, returns NSE.
    """
    base_cfg = FrozenDict(base_config_dict)
    model_name = base_cfg["model"]["name"]
    has_building = "building" in base_cfg

    # --- 1. Define Hyperparameter Search Space ---
    trial_params = {} # Store suggested params

    # === Training Hyperparameters ===
    trial_params["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    trial_params["batch_size"] = trial.suggest_categorical("batch_size", [256, 512, 1024])
    # Epochs are fixed via run_optimization.py --opt_epochs argument

    # === Model Hyperparameters ===
    trial_params["model_width"] = trial.suggest_categorical("model_width", [256, 512, 1024])
    trial_params["model_depth"] = trial.suggest_int("model_depth", 3, 6)

    if model_name == "FourierPINN":
        trial_params["ff_dims"] = trial.suggest_categorical("ff_dims", [128, 256, 512])
        trial_params["fourier_scale"] = trial.suggest_float("fourier_scale", 5.0, 20.0)
    elif model_name == "SIREN":
        trial_params["w0"] = trial.suggest_float("w0", 1.0, 30.0)
    # Add parameters for DGMNetwork if needed

    # === Grid Hyperparameters ===
    # (Same as before - these are independent of data_free/gradnorm)
    nx_base = trial.suggest_int("nx_base", 20, 80, step=4)
    ny_base = trial.suggest_int("ny_base", 10, 40, step=2)
    nt_base = trial.suggest_int("nt_base", 10, 40, step=2)
    trial_params["grid"] = {"nx": nx_base, "ny": ny_base, "nt": nt_base}
    trial_params["ic_bc_grid"] = {
        "nx_ic": max(5, int(trial.suggest_float("nx_ic_factor", 0.5, 1.5) * nx_base)),
        "ny_ic": max(5, int(trial.suggest_float("ny_ic_factor", 0.5, 1.5) * ny_base)),
        "ny_bc_left": max(5, int(trial.suggest_float("ny_bc_left_factor", 0.3, 1.2) * ny_base)),
        "nt_bc_left": max(5, int(trial.suggest_float("nt_bc_left_factor", 0.3, 1.2) * nt_base)),
        "ny_bc_right": max(5, int(trial.suggest_float("ny_bc_right_factor", 0.3, 1.2) * ny_base)),
        "nt_bc_right": max(5, int(trial.suggest_float("nt_bc_right_factor", 0.3, 1.2) * nt_base)),
        "nx_bc_bottom": max(5, int(trial.suggest_float("nx_bc_bottom_factor", 0.5, 1.5) * nx_base)),
        "nt_bc_other": max(5, int(trial.suggest_float("nt_bc_other_factor", 0.3, 1.2) * nt_base)),
        "nx_bc_top": max(5, int(trial.suggest_float("nx_bc_top_factor", 0.5, 1.5) * nx_base)),
    }
    if has_building:
         trial_params["building_grid"] = {
             "nx": max(5, int(trial.suggest_float("nx_bldg_factor", 0.3, 1.2) * nx_base)),
             "ny": max(5, int(trial.suggest_float("ny_bldg_factor", 0.3, 1.2) * ny_base)),
             "nt": max(5, int(trial.suggest_float("nt_bldg_factor", 0.3, 1.2) * nt_base)),
         }

    # === Loss Weights / GradNorm Hyperparameters (Conditional Suggestion) ===
    trial_params["loss_weights"] = {} # Initialize weights dict

    if enable_gradnorm:
        print(f"Trial {trial.number}: Configuring for GradNorm (data_free={data_free}).") # Debug print
        # *** Suggest GradNorm specific parameters ***
        trial_params["gradnorm_alpha"] = trial.suggest_float("gradnorm_alpha", 0.1, 3.0)
        trial_params["gradnorm_update_freq"] = trial.suggest_categorical("gradnorm_update_freq", [50, 100, 200, 500])
        # trial_params["gradnorm_lr"] = trial.suggest_float("gradnorm_lr", 1e-3, 1e-1, log=True) # Optional

        # Set initial weights to 1.0; GradNorm adjusts them
        trial_params["loss_weights"]["pde_weight"] = 1.0
        trial_params["loss_weights"]["ic_weight"] = 1.0
        trial_params["loss_weights"]["bc_weight"] = 1.0
        if has_building: trial_params["loss_weights"]["building_bc_weight"] = 1.0

        # Data weight is 1.0 ONLY if GradNorm is ON *and* data_free is OFF
        trial_params["loss_weights"]["data_weight"] = 1.0 if not data_free else 0.0

        # Set irrelevant static factors to None for Optuna logging clarity
        trial.set_user_attr("ic_weight_factor", None)
        trial.set_user_attr("bc_weight_factor", None)
        if has_building: trial.set_user_attr("building_bc_weight_factor", None)
        trial.set_user_attr("data_weight_factor", None) # Even if data_free=False

    else: # Static weights mode
        print(f"Trial {trial.number}: Configuring static weights (data_free={data_free}).") # Debug print
        # *** Suggest static weight factors ***
        trial_params["loss_weights"]["pde_weight"] = 1.0 # Fixed reference for static weights
        # Suggest factors, calculate absolute weights (relative to PDE=1.0)
        ic_factor = trial.suggest_float("ic_weight_factor", 1e-3, 1e3, log=True)
        bc_factor = trial.suggest_float("bc_weight_factor", 1e-3, 1e3, log=True)
        trial_params["loss_weights"]["ic_weight"] = ic_factor * trial_params["loss_weights"]["pde_weight"]
        trial_params["loss_weights"]["bc_weight"] = bc_factor * trial_params["loss_weights"]["pde_weight"]
        if has_building:
            bldg_factor = trial.suggest_float("building_bc_weight_factor", 1e-3, 1e3, log=True)
            trial_params["loss_weights"]["building_bc_weight"] = bldg_factor * trial_params["loss_weights"]["pde_weight"]

        # Data weight factor only optimized if not data_free
        if not data_free:
            data_factor = trial.suggest_float("data_weight_factor", 1e-3, 1e3, log=True)
            trial_params["loss_weights"]["data_weight"] = data_factor * trial_params["loss_weights"]["pde_weight"]
        else:
            trial_params["loss_weights"]["data_weight"] = 0.0 # Explicitly zero if data_free
            trial.set_user_attr("data_weight_factor", None) # Log as None if not used

        # Set irrelevant GradNorm params to None for Optuna logging clarity
        trial.set_user_attr("gradnorm_alpha", None)
        trial.set_user_attr("gradnorm_update_freq", None)
        # trial.set_user_attr("gradnorm_lr", None) # If you were suggesting it

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
         if "building" not in trial_config_dict: trial_config_dict["building"] = {}
         trial_config_dict["building"]["nx"] = trial_params["building_grid"]["nx"]
         trial_config_dict["building"]["ny"] = trial_params["building_grid"]["ny"]
         trial_config_dict["building"]["nt"] = trial_params["building_grid"]["nt"]

    # --- Update loss weights and GradNorm config section based on enable_gradnorm flag ---
    trial_config_dict["loss_weights"] = trial_params["loss_weights"]

    # Ensure gradnorm key exists
    if "gradnorm" not in trial_config_dict: trial_config_dict["gradnorm"] = {}

    if enable_gradnorm:
        trial_config_dict["gradnorm"]["enable"] = True
        trial_config_dict["gradnorm"]["alpha"] = trial_params["gradnorm_alpha"]
        trial_config_dict["gradnorm"]["update_freq"] = trial_params["gradnorm_update_freq"]
        # Use base LR or the suggested one
        trial_config_dict["gradnorm"]["learning_rate"] = trial_params.get("gradnorm_lr", base_cfg.get("gradnorm",{}).get("learning_rate", 0.01))
    else:
        # Explicitly disable GradNorm in the trial config if it's not enabled
        trial_config_dict["gradnorm"]["enable"] = False
        # Clean up keys that might have been in base_config but are irrelevant now
        trial_config_dict["gradnorm"].pop("alpha", None)
        trial_config_dict["gradnorm"].pop("update_freq", None)
        # trial_config_dict["gradnorm"].pop("learning_rate", None) # Decide if you want base LR here

    # Convert final config to FrozenDict for JAX functions
    trial_cfg_frozen = FrozenDict(trial_config_dict)

    # --- Run the training trial ---
    try:
        # Pass the Optuna trial object for pruning AND the data_free flag
        best_nse = run_training_trial(trial, trial_cfg_frozen, data_free) # Pass data_free flag HERE

        # Ensure a standard float is returned for Optuna
        if isinstance(best_nse, (jax.Array, jnp.ndarray, np.ndarray)):
            best_nse = float(best_nse)

        if jnp.isnan(best_nse) or not isinstance(best_nse, (float, int)) or best_nse <= -float('inf'):
             print(f"Trial {trial.number}: Invalid NSE ({best_nse}). Returning -1.0.")
             return -1.0 # Optuna prefers finite numbers

        return best_nse

    except optuna.exceptions.TrialPruned as e:
         raise e # Re-raise for Optuna to handle pruning correctly
    except Exception as e:
         print(f"Trial {trial.number}: UNHANDLED EXCEPTION in run_training_trial: {e}")
         import traceback
         traceback.print_exc()
         return -1.0 # Indicate failure with a poor score