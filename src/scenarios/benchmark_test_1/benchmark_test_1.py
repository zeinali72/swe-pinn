import os
import sys
import time
import copy
import argparse
import importlib
import itertools
from typing import Any, Dict, Tuple
import shutil

import jax
import jax.numpy as jnp
from jax import random, lax
import optax
from aim import Repo, Run, Image
from flax.core import FrozenDict
import numpy as np 
import matplotlib.pyplot as plt 

# Local application imports
from src.config import load_config, DTYPE
from src.data import (
    sample_domain, 
    get_batches_tensor,
    get_sample_count,
    bathymetry_fn,
    load_boundary_condition,
    load_bathymetry 
)
from src.models import init_model
from src.losses import (
    compute_pde_loss,
    loss_boundary_dirichlet_h,
    loss_boundary_wall_horizontal,
    loss_boundary_wall_vertical,
    total_loss
)
from src.utils import ( 
   nse, rmse, generate_trial_name, save_model, ask_for_confirmation
)

from src.reporting import (
    print_epoch_stats, log_metrics, print_final_summary
)


def train_step(
        model: Any, 
        optimiser: optax.GradientTransformation, 
        params: FrozenDict, 
        opt_state: optax.OptState, 
        batch: Dict[str, jnp.ndarray], 
        config: Dict[str, Any], 
        bc_fn_static: Any,
        weights_dict: FrozenDict # Type hint updated
        ) -> Tuple[FrozenDict, optax.OptState, Dict[str, float], float]:
    """
    Performs one step of gradient descent.
    """
    
    # weights_dict is now a FrozenDict (hashable), so .keys() works fine
    active_loss_keys_base = list(weights_dict.keys())

    def loss_fn(params):
        
        terms = {}
        # --- 1. PDE Loss (Physics + Bathymetry) ---
        loss_pde = compute_pde_loss(model, params, batch['pde'], config)
        
        # --- 2. Initial Condition Loss (t=0, h=9.7) ---
        U_ic = model.apply({'params': params}, batch['ic'], train=True)
        h_ic_pred = U_ic[..., 0]
        hu_ic_pred = U_ic[..., 1]
        hv_ic_pred = U_ic[..., 2]

        # Get bathymetry at IC points
        z_ic, _, _ = bathymetry_fn(batch['ic'][..., 0], batch['ic'][..., 1])
        
        # Calculate target depth based on absolute water level 9.7m
        # h_target = max(0, 9.7 - z)
        h_target_ic = jnp.maximum(0.0, 9.7 - z_ic)

        loss_ic_h = jnp.mean((h_ic_pred - h_target_ic)**2)
        # Enforce zero velocity at t=0
        loss_ic_vel = jnp.mean(hu_ic_pred**2 + hv_ic_pred**2) 
        loss_ic = loss_ic_h + loss_ic_vel

        # --- 3. Boundary Conditions ---
        
        # A. Left Boundary (x=0): Time-Varying Water Level
        t_left = batch['bc_left'][..., 2]
        bc_level_abs = bc_fn_static(t_left) # Interpolate target Absolute Level
        
        # Get Z at boundary to calculate depth h = Level - Z
        z_left, _, _ = bathymetry_fn(batch['bc_left'][..., 0], batch['bc_left'][..., 1])
        h_target_left = jnp.maximum(0.0, bc_level_abs - z_left)
        
        loss_bc_left = loss_boundary_dirichlet_h(model, {'params': params}, batch['bc_left'], h_target_left)
        
        # B. Right Boundary (x=700): Slip Walls (No flux x)
        loss_bc_right = loss_boundary_wall_vertical(model, {'params': params}, batch['bc_right'])
        
        # C. Top & Bottom Boundaries (y=0, y=100): Slip Walls (No flux y)
        loss_bc_top = loss_boundary_wall_horizontal(model, {'params': params}, batch['bc_top'])
        loss_bc_bottom = loss_boundary_wall_horizontal(model, {'params': params}, batch['bc_bottom'])
        
        total_bc = loss_bc_left + loss_bc_right + loss_bc_top + loss_bc_bottom

        terms = {
            'pde': loss_pde,
            'ic': loss_ic,
            'bc': total_bc # Renamed to 'bc' to match weights keys if needed, or 'total_bc'
        }

        # --- 4. Weighted Sum ---
        # Helper to safely get term or 0.0
        terms_with_defaults = {k: terms.get(k, 0.0) for k in weights_dict.keys()}
        total = total_loss(terms_with_defaults, weights_dict)
        
        return total, terms

    # Calculate Gradients
    (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Update Parameters
    updates, new_opt_state = optimiser.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, metrics, loss_val

# JIT Compile
train_step_jitted = jax.jit(train_step, static_argnames=['model', 'optimiser', 'config', 'bc_fn_static', 'weights_dict'])

def main(config_path: str):
    """
    Main training loop for Benchmark Test 1 scenario.
    """
    
    #--- 1. LOAD CONFIGURATION ---
    cfg_dict = load_config(config_path)
    cfg = FrozenDict(cfg_dict)

    print("Info: Running Benchmark Test 1 Scenario model training...")

    try:
        models_module = importlib.import_module("src.models")
        model_class = getattr(models_module, cfg["model"]["name"])
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not find model class '{cfg['model']['name']}' in src/models.py") from e
    
    key = random.PRNGKey(cfg["training"]["seed"])
    model_key, train_key, val_key = random.split(key, 3)
    model, params = init_model(model_class, model_key, cfg)

    # --- 2. Setup Directories ---
    config_base = os.path.splitext(os.path.basename(cfg['CONFIG_PATH']))[0]
    trial_name = generate_trial_name(config_base)
    results_dir = os.path.join("results", trial_name)
    model_dir = os.path.join("models", trial_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --- 3. Setup Optimizer ---
    reduce_on_plateau_cfg = cfg.get("training", {}).get("reduce_on_plateau", {})
    optimiser = optax.chain(
        optax.clip_by_global_norm(cfg.get("training", {}).get("clip_norm", 1.0)),
        optax.adam(learning_rate=cfg["training"]["learning_rate"]),
        optax.contrib.reduce_on_plateau(
            factor=float(reduce_on_plateau_cfg.get("factor", 0.5)),
            patience=int(reduce_on_plateau_cfg.get("patience", 5)),
            rtol=float(reduce_on_plateau_cfg.get("rtol", 1e-4)),
            atol=float(reduce_on_plateau_cfg.get("atol", 0.0)),
            cooldown=int(reduce_on_plateau_cfg.get("cooldown", 1)),
            accumulation_size=int(reduce_on_plateau_cfg.get("accumulation_size", 235)),
            min_scale=float(reduce_on_plateau_cfg.get("min_scale", 1e-6)),
        ),
    )
    opt_state = optimiser.init(params)

    # --- 4. Load Data Assets ---
    scenario_name = cfg.get('scenario')
    if not scenario_name:
         print(f"Error: 'scenario' key must be set in config '{config_path}'.")
         sys.exit(1)
         
    base_data_path = os.path.join("data", scenario_name)
    
    # A. Load Bathymetry (REQUIRED)
    dem_path = os.path.join(base_data_path, "test1DEM.asc")
    if not os.path.exists(dem_path):
        print(f"Error: DEM file not found at {dem_path}")
        sys.exit(1)
    print(f"Loading Bathymetry from {dem_path}...")
    load_bathymetry(dem_path)

    # B. Load Boundary Condition Function
    bc_csv_path = os.path.join(base_data_path, "Test1BC.csv")
    if not os.path.exists(bc_csv_path):
        print(f"Error: Boundary condition CSV file not found at {bc_csv_path}.")
        sys.exit(1)
    bc_fn_static = load_boundary_condition(bc_csv_path)

    # C. Load Validation Data (Optional)
    val_points, h_true_val = None, None
    validation_data_file = os.path.join(base_data_path, "validation_sample.npy")
    validation_data_loaded = False
    
    if os.path.exists(validation_data_file):
        try:
            print(f"Loading VALIDATION data from: {validation_data_file}")
            loaded_val_data = jnp.load(validation_data_file).astype(DTYPE)
            val_points_all = loaded_val_data[:, [1, 2, 0]]
            h_true_val_all = loaded_val_data[:, 3]
            num_val_points = val_points_all.shape[0]
            if num_val_points > 0:
                validation_data_loaded = True
        except Exception as e:
            print(f"Warning: Error loading validation data: {e}")
    else:
        print(f"Warning: Validation data not found. Skipping dense validation.")

    # --- 5. Prepare Loss Weights ---
    static_weights_dict = {k.replace('_weight',''):v for k,v in cfg["loss_weights"].items()}
    active_loss_term_keys = [k for k, v in static_weights_dict.items() if v > 0]
    
    # FIX: Convert to FrozenDict so it is Hashable for JAX Static Args
    current_weights_dict = FrozenDict({k: static_weights_dict[k] for k in active_loss_term_keys})

    # --- 6. Initialize Aim & Log Source Code ---
    aim_repo = None
    aim_run = None
    run_hash = None
    try:
        aim_repo_path = "aim_repo"
        if not os.path.exists(aim_repo_path):
             os.makedirs(aim_repo_path, exist_ok=True)
        aim_repo = Repo(path=aim_repo_path, init=True)
        aim_run = Run(repo=aim_repo, experiment=trial_name)
        run_hash = aim_run.hash
        
        # Log basics
        hparams_to_log = copy.deepcopy(cfg_dict)
        aim_run["hparams"] = hparams_to_log
        aim_run['flags'] = {"scenario_type": "benchmark_test_1"}
        
        # --- Log Config and Script as Artifacts ---
        try:
            aim_run.log_artifact(config_path, name='run_config.yaml')
            current_script_path = os.path.abspath(__file__)
            aim_run.log_artifact(current_script_path, name='source_script.py')
            print("Logged config and source script to Aim.")
        except Exception as e_art: 
            print(f"Warning: Failed to log initial artifacts: {e_art}")
            
        print(f"Aim tracking initialized: {trial_name} ({run_hash})")
    except Exception as e:
        print(f"Warning: Aim tracking failed to initialize: {e}")

    # --- 7. Summary ---
    print(f"\n--- Training Started: {trial_name} ---")
    print(f"Model: {cfg['model']['name']}, Epochs: {cfg['training']['epochs']}")
    print(f"Active Loss Terms: {active_loss_term_keys}")

    # --- 8. Data Generation Setup ---
    sampling_cfg = cfg["sampling"]
    batch_size = cfg["training"]["batch_size"]
    domain_cfg = cfg["domain"]
    
    n_pde = get_sample_count(sampling_cfg, "n_points_pde", 1000)
    n_ic = get_sample_count(sampling_cfg, "n_points_ic", 100)
    n_bc_domain = get_sample_count(sampling_cfg, "n_points_bc_domain", 100)
    n_bc_per_wall = max(5, n_bc_domain // 4)

    # Check batch size viability
    bc_counts = [n_pde//batch_size, n_ic//batch_size, n_bc_per_wall//batch_size]
    num_batches = max(bc_counts) if bc_counts else 0
    
    if num_batches == 0:
        print(f"Error: Batch size {batch_size} is too large for sample counts.")
        return -1.0
    print(f"Batches per epoch: {num_batches}")

    # JIT Data Generator
    def generate_epoch_data(key):
        key, pde_key, ic_key, bc_keys = random.split(key, 4)
        
        # PDE
        if n_pde // batch_size > 0:
            pde_pts = sample_domain(pde_key, n_pde, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"]))
            pde_data = get_batches_tensor(pde_key, pde_pts, batch_size, num_batches)
        else:
            pde_data = jnp.zeros((num_batches, 0, 3), dtype=DTYPE)

        # IC
        if n_ic // batch_size > 0:
            ic_pts = sample_domain(ic_key, n_ic, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., 0.))
            ic_data = get_batches_tensor(ic_key, ic_pts, batch_size, num_batches)
        else:
            ic_data = jnp.zeros((num_batches, 0, 3), dtype=DTYPE)
            
        # BCs
        l_key, r_key, b_key, t_key = random.split(bc_keys, 4)
        def get_wall(k, n, x_b, y_b):
            if n // batch_size > 0:
                pts = sample_domain(k, n, x_b, y_b, (0., domain_cfg["t_final"]))
                return get_batches_tensor(k, pts, batch_size, num_batches)
            return jnp.zeros((num_batches, 0, 3), dtype=DTYPE)

        bc_left = get_wall(l_key, n_bc_per_wall, (0., 0.), (0., domain_cfg["ly"]))
        bc_right = get_wall(r_key, n_bc_per_wall, (domain_cfg["lx"], domain_cfg["lx"]), (0., domain_cfg["ly"]))
        bc_bot = get_wall(b_key, n_bc_per_wall, (0., domain_cfg["lx"]), (0., 0.))
        bc_top = get_wall(t_key, n_bc_per_wall, (0., domain_cfg["lx"]), (domain_cfg["ly"], domain_cfg["ly"]))

        return {
            'pde': pde_data, 'ic': ic_data,
            'bc': {'left': bc_left, 'right': bc_right, 'bottom': bc_bot, 'top': bc_top}
        }
    
    generate_epoch_data_jitted = jax.jit(generate_epoch_data)

    # Scan Body
    def scan_body(carry, batch_data):
        curr_params, curr_opt_state = carry
        
        current_all_batches = {
            'pde': batch_data['pde'],
            'ic': batch_data['ic'],
            'bc_left': batch_data['bc']['left'],
            'bc_right': batch_data['bc']['right'],
            'bc_bottom': batch_data['bc']['bottom'],
            'bc_top': batch_data['bc']['top']
        }

        # current_weights_dict is now FrozenDict, so it's hashable
        new_params, new_opt_state, terms, total = train_step_jitted(
            model, optimiser, curr_params, curr_opt_state,
            current_all_batches, cfg, bc_fn_static, current_weights_dict
        )
        return (new_params, new_opt_state), (terms, total)

    # --- 9. Training Loop ---
    best_nse_stats = {'nse': -jnp.inf, 'epoch': 0}
    best_loss_stats = {'total_weighted_loss': jnp.inf, 'epoch': 0}
    best_params_nse = None
    best_params_loss = None # Added tracking for best loss model
    
    start_time = time.time()
    global_step = 0

    try:
        for epoch in range(cfg["training"]["epochs"]):
            epoch_start_time = time.time()

            # Generate Data & Train
            train_key, epoch_key = random.split(train_key)
            scan_inputs = generate_epoch_data_jitted(epoch_key)
            
            (params, opt_state), (batch_losses, batch_totals) = lax.scan(
                scan_body, (params, opt_state), scan_inputs
            )
            global_step += num_batches

            # Average Losses
            avg_losses = {k: float(jnp.sum(v))/num_batches for k, v in batch_losses.items()}
            avg_total = float(jnp.sum(batch_totals))/num_batches

            # Validation
            nse_val, rmse_val = -jnp.inf, jnp.inf
            if validation_data_loaded:
                try:
                    U_val = model.apply({'params': params}, val_points_all, train=False)
                    nse_val = nse(h_true_val_all, U_val[..., 0])
                    rmse_val = rmse(h_true_val_all, U_val[..., 0])
                except: pass

            # Update Best NSE
            if nse_val > best_nse_stats['nse']:
                best_nse_stats.update({'nse': nse_val, 'epoch': epoch})
                best_params_nse = copy.deepcopy(params)
                print(f"    ---> New best NSE: {nse_val:.6f}")
            
            # Update Best Loss (Tracking)
            if avg_total < best_loss_stats['total_weighted_loss']:
                best_loss_stats.update({'total_weighted_loss': avg_total, 'epoch': epoch})
                best_params_loss = copy.deepcopy(params)

            # Logging
            if (epoch + 1) % 100 == 0:
                print_epoch_stats(epoch, global_step, start_time, avg_total, 
                                  avg_losses.get('pde',0), avg_losses.get('ic',0), avg_losses.get('bc',0),
                                  0, 0, 0, nse_val, rmse_val, time.time()-epoch_start_time)

                if aim_run:
                    log_metrics(aim_run, global_step, epoch, {
                        'losses': avg_losses, 'total': avg_total, 'val': {'nse': nse_val}
                    })

    except KeyboardInterrupt:
        print("\n--- Training interrupted ---")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    # --- 10. Post-Training (Save & Plot) ---
    finally:
        total_time = time.time() - start_time
        print_final_summary(total_time, best_nse_stats, best_loss_stats)

        # Decide which params to save (NSE preferred if available, else Loss)
        final_params = best_params_nse if best_params_nse is not None else best_params_loss

        if ask_for_confirmation():
            if final_params is not None:
                # Capture the path where the model is saved locally
                saved_model_path = save_model(final_params, model_dir, trial_name)
                
                # --- NEW: Log Model as Artifact to Aim ---
                if aim_run and saved_model_path:
                    try:
                        aim_run.log_artifact(saved_model_path, name='best_model.pkl')
                        print(f"Logged model artifact to Aim.")
                    except Exception as e_mod:
                        print(f"Warning: Failed to log model artifact: {e_mod}")
                
                # --- Plotting Specific to Test 1 ---
                print("Generating Test 1 plots...")
                t_plot = jnp.arange(0., cfg['domain']['t_final'], 60.0, dtype=DTYPE)
                
                def plot_gauge(x, y, name, color, filename):
                    pts = jnp.stack([jnp.full_like(t_plot, x), jnp.full_like(t_plot, y), t_plot], axis=-1)
                    U = model.apply({'params': final_params}, pts, train=False)
                    h_pred = U[..., 0]
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(t_plot, h_pred, label=f'Predicted h @ ({x},{y})', color=color)
                    plt.xlabel('Time (s)')
                    plt.ylabel('Water Level h (m)')
                    plt.title(f'{name} - Water Level vs Time')
                    plt.legend()
                    plt.grid(True)
                    path = os.path.join(results_dir, filename)
                    plt.savefig(path)
                    plt.close()
                    if aim_run:
                        aim_run.log_image(Image(filename=path), name=filename)

                plot_gauge(400.0, 50.0, "Point 1", "blue", "P1_timeseries.png")
                plot_gauge(600.0, 50.0, "Point 2", "red",  "P2_timeseries.png")
                print(f"Plots saved to {results_dir}")
            else:
                print("No model parameters found to save.")

        if aim_run: aim_run.close()

    return best_nse_stats['nse']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified PINN training script for SWE (Test 1).")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path: sys.path.insert(0, project_root)

    main(args.config)