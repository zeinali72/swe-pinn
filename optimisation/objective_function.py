# optimisation/objective_function.py
"""
Defines the Optuna objective function: suggests hyperparameters based on flags,
builds the trial configuration, stores it, and calls the training loop.
"""
import jax
import optuna
from flax.core import FrozenDict
import numpy as np
import jax.numpy as jnp
from typing import Dict, Any
import copy
import yaml

# Import the training loop function
from optimisation.optimization_train_loop import run_training_trial

def objective(trial: optuna.trial.Trial,
              base_config_dict: Dict,
              data_free: bool,           # Flag passed from run_optimization
              enable_gradnorm: bool) -> float: # Flag passed from run_optimization
    """
    Objective function for Optuna HPO.
    """
    base_cfg = FrozenDict(base_config_dict) # Use FrozenDict internally if needed by functions
    model_name = base_cfg["model"]["name"]
    has_building = "building" in base_cfg

    # --- 1. Define Hyperparameter Search Space ---
    trial_params = {} # Store suggested params

    # === Training Hyperparameters ===
    trial_params["learning_rate"] = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    trial_params["batch_size"] = trial.suggest_categorical("batch_size", [256, 512, 1024])
    # opt_epochs comes from base_cfg['training']['opt_epochs']

    # === Model Hyperparameters ===
    trial_params["model_width"] = trial.suggest_categorical("model_width", [256, 512, 1024])
    trial_params["model_depth"] = trial.suggest_int("model_depth", 3, 6)
    if model_name == "FourierPINN":
        trial_params["ff_dims"] = trial.suggest_categorical("ff_dims", [128, 256, 512])
        trial_params["fourier_scale"] = trial.suggest_float("fourier_scale", 5.0, 20.0)

    # === Grid Hyperparameters (MODIFIED AS REQUESTED) ===
    # 1. PDE (Base)
    nx_base = trial.suggest_int("nx_base", 20, 80, step=4)
    ny_base = trial.suggest_int("ny_base", 10, 40, step=2)
    nt_base = trial.suggest_int("nt_base", 10, 40, step=2)
    trial_params["grid"] = {"nx": nx_base, "ny": ny_base, "nt": nt_base}

    # 2. IC (One factor)
    ic_factor = trial.suggest_float("ic_factor", 0.2, 1.5)
    
    # 3. Left BC (Inflow)
    left_bc_factor = trial.suggest_float("left_bc_factor", 0.2, 1.5)
    
    # 4. Right BC (Outlet)
    right_bc_factor = trial.suggest_float("right_bc_factor", 0.2, 1.5)

    # 5. Horizontal BCs (Top/Bottom Walls)
    horizontal_bc_factor = trial.suggest_float("horizontal_bc_factor", 0.2, 1.5)
    
    trial_params["ic_bc_grid"] = {
        # Use ic_factor for both nx_ic and ny_ic
        "nx_ic": max(5, int(ic_factor * nx_base)),
        "ny_ic": max(5, int(ic_factor * ny_base)),
        
        # Use left_bc_factor for both ny_bc_left and nt_bc_left
        "ny_bc_left": max(5, int(left_bc_factor * ny_base)),
        "nt_bc_left": max(5, int(left_bc_factor * nt_base)),
        
        # Use right_bc_factor for both ny_bc_right and nt_bc_right
        "ny_bc_right": max(5, int(right_bc_factor * ny_base)),
        "nt_bc_right": max(5, int(right_bc_factor * nt_base)),
        
        # Use horizontal_bc_factor for top and bottom (nx and nt)
        "nx_bc_bottom": max(5, int(horizontal_bc_factor * nx_base)),
        "nt_bc_other": max(5, int(horizontal_bc_factor * nt_base)), # This name is from the original config
        "nx_bc_top": max(5, int(horizontal_bc_factor * nx_base)),
    }
    
    if has_building:
         # 6. Building BCs (One factor)
         building_bc_factor = trial.suggest_float("building_bc_factor", 0.2, 1.5)
         trial_params["building_grid"] = {
             "nx": max(5, int(building_bc_factor * nx_base)),
             "ny": max(5, int(building_bc_factor * ny_base)),
             "nt": max(5, int(building_bc_factor * nt_base)),
         }

    # === Loss Weights / GradNorm Hyperparameters (Conditional Suggestion) ===
    trial_params["loss_weights"] = {} # Initialize weights dict

    if enable_gradnorm:
        print(f"Trial {trial.number}: Configuring for GradNorm (data_free={data_free}).")
        trial_params["gradnorm_alpha"] = trial.suggest_float("gradnorm_alpha", 0.1, 3.0)
        trial_params["gradnorm_update_freq"] = trial.suggest_categorical("gradnorm_update_freq", [50, 100, 200, 500])
        trial_params["gradnorm_lr"] = trial.suggest_float("gradnorm_lr", 1e-3, 1e-1, log=True)

        # Set initial weights to 1.0; GradNorm adjusts them
        trial_params["loss_weights"]["pde_weight"] = 1.0
        trial_params["loss_weights"]["ic_weight"] = 1.0
        trial_params["loss_weights"]["bc_weight"] = 1.0
        if has_building: trial_params["loss_weights"]["building_bc_weight"] = 1.0
        trial_params["loss_weights"]["data_weight"] = 1.0 if not data_free else 0.0 # Set based on data_free flag

        # Log irrelevant static factors as None
        trial.set_user_attr("ic_weight_factor", None)
        trial.set_user_attr("bc_weight_factor", None)
        if has_building: trial.set_user_attr("building_bc_weight_factor", None)
        trial.set_user_attr("data_weight_factor", None)

    else: # Static weights mode
        print(f"Trial {trial.number}: Configuring static weights (data_free={data_free}).")
        trial_params["loss_weights"]["pde_weight"] = 1.0 # Fixed reference
        # Suggest factors, calculate absolute weights
        ic_factor = trial.suggest_float("ic_weight_factor", 1e-4, 1e3, log=True)
        bc_factor = trial.suggest_float("bc_weight_factor", 1e-4, 1e3, log=True)
        trial_params["loss_weights"]["ic_weight"] = ic_factor * trial_params["loss_weights"]["pde_weight"]
        trial_params["loss_weights"]["bc_weight"] = bc_factor * trial_params["loss_weights"]["pde_weight"]
        if has_building:
            bldg_factor = trial.suggest_float("building_bc_weight_factor", 1e-4, 1e3, log=True)
            trial_params["loss_weights"]["building_bc_weight"] = bldg_factor * trial_params["loss_weights"]["pde_weight"]

        if not data_free:
            data_factor = trial.suggest_float("data_weight_factor", 1e-4, 1e3, log=True)
            trial_params["loss_weights"]["data_weight"] = data_factor * trial_params["loss_weights"]["pde_weight"]
        else:
            trial_params["loss_weights"]["data_weight"] = 0.0 # Explicitly zero
            trial.set_user_attr("data_weight_factor", None) # Log as None

        # Log irrelevant GradNorm params as None
        trial.set_user_attr("gradnorm_alpha", None)
        trial.set_user_attr("gradnorm_update_freq", None)
        trial.set_user_attr("gradnorm_lr", None)

    # === Construct FULL Trial Configuration Dictionary ===
    # Start with a deep copy of the base config (which is already a dict)
    trial_config_dict = copy.deepcopy(base_config_dict)

    # --- Update sections with suggested hyperparameters ---
    trial_config_dict["training"]["learning_rate"] = trial_params["learning_rate"]
    trial_config_dict["training"]["batch_size"] = trial_params["batch_size"]
    # 'opt_epochs' is already in trial_config_dict['training'] from run_optimization.py

    trial_config_dict["model"]["width"] = trial_params["model_width"]
    trial_config_dict["model"]["depth"] = trial_params["model_depth"]
    if model_name == "FourierPINN":
        trial_config_dict["model"]["ff_dims"] = trial_params["ff_dims"]
        trial_config_dict["model"]["fourier_scale"] = trial_params["fourier_scale"]

    trial_config_dict["grid"] = trial_params["grid"]
    trial_config_dict["ic_bc_grid"] = trial_params["ic_bc_grid"]
    if has_building:
         if "building" not in trial_config_dict: trial_config_dict["building"] = {}
         trial_config_dict["building"]["nx"] = trial_params["building_grid"]["nx"]
         trial_config_dict["building"]["ny"] = trial_params["building_grid"]["ny"]
         trial_config_dict["building"]["nt"] = trial_params["building_grid"]["nt"]

    # Update loss weights (already contains suggested values)
    trial_config_dict["loss_weights"] = trial_params["loss_weights"]

    # Update GradNorm config section based on the enable_gradnorm flag
    if "gradnorm" not in trial_config_dict: trial_config_dict["gradnorm"] = {}
    if enable_gradnorm:
        trial_config_dict["gradnorm"]["enable"] = True
        trial_config_dict["gradnorm"]["alpha"] = trial_params["gradnorm_alpha"]
        trial_config_dict["gradnorm"]["update_freq"] = trial_params["gradnorm_update_freq"]
        trial_config_dict["gradnorm"]["learning_rate"] = trial_params.get("gradnorm_lr", base_cfg.get("gradnorm",{}).get("learning_rate", 0.01))
    else:
        trial_config_dict["gradnorm"]["enable"] = False
        # Clean up irrelevant keys
        trial_config_dict["gradnorm"].pop("alpha", None)
        trial_config_dict["gradnorm"].pop("update_freq", None)

    # --- Store the complete configuration in user attributes ---
    # Convert back to regular dict for storage if needed, ensure serializability
    config_to_store = dict(trial_config_dict)
    trial.set_user_attr('full_config', config_to_store)

    # --- Convert final config to FrozenDict for JAX functions ---
    trial_cfg_frozen = FrozenDict(trial_config_dict)

# <<<--- START: ADD PRINT STATEMENTS HERE --- >>>
    print("-" * 50)
    print(f"Starting Trial {trial.number}")
    print("Suggested Hyperparameters:")
    # Print the parameters Optuna actually suggested for this trial
    for key, value in trial.params.items():
         # Check if the parameter exists (it might be None if not suggested in that trial branch)
         if value is not None:
             if isinstance(value, float):
                  print(f"  {key:<25}: {value:.6e}" if abs(value) < 1e-2 or abs(value) > 1e3 else f"  {key:<25}: {value:.6f}")
             else:
                  print(f"  {key:<25}: {value}")
         # else: You could optionally print "Not suggested" here too

    print("\nFull Configuration for this Trial:")
    # Use yaml.dump for a readable, multi-line output of the dictionary
    # Make sure to use the regular dict (config_to_store), not the FrozenDict
    print(yaml.dump(config_to_store, default_flow_style=False, sort_keys=False, indent=2))
    print("-" * 50)
    # <<<--- END: ADD PRINT STATEMENTS HERE --- >>>

    # --- Run the training trial ---
    try:
        # Pass the Optuna trial object, the FULL config, and the data_free flag
        best_nse = run_training_trial(trial, trial_cfg_frozen, data_free)

        # Ensure a standard float is returned
        if isinstance(best_nse, (jax.Array, jnp.ndarray, np.ndarray)):
            best_nse = float(best_nse)

        if jnp.isnan(best_nse) or not isinstance(best_nse, (float, int)) or best_nse <= -float('inf'):
             print(f"Trial {trial.number}: Invalid NSE ({best_nse}). Returning -1.0.")
             return -1.0

        return best_nse

    except optuna.exceptions.TrialPruned as e:
         raise e # Re-raise for Optuna
    except Exception as e:
         print(f"Trial {trial.number}: UNHANDLED EXCEPTION in run_training_trial: {e}")
         import traceback
         traceback.print_exc()
         return -1.0 # Indicate failure