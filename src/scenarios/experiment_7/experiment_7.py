import os
import sys
import time
import copy
import argparse
import importlib
import itertools
from typing import Any, Dict, Tuple
import shutil
import pandas as pd 

from jaxtyping import config

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
    get_batches_tensor,
    get_sample_count,
    load_boundary_condition,
    IrregularDomainSampler,
    load_bathymetry
)
from src.models import init_model
from src.losses import (
    compute_pde_loss,
    loss_boundary_dirichlet_hu,
    loss_boundary_dirichlet_hv,
    loss_slip_wall_generalized,
    compute_neg_h_loss,
    compute_data_loss,
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
        data_free: bool,
        bc_fn_static: Any,
        weights_dict: FrozenDict 
        ) -> Tuple[FrozenDict, optax.OptState, Dict[str, float], float]:
    
    active_loss_keys_base = list(weights_dict.keys())

    def loss_fn(params):
        
        # --- 1. PDE Loss ---
        loss_pde = compute_pde_loss(model, params, batch['pde'], config)
        loss_neg_h = compute_neg_h_loss(model, params, batch['pde'])
        
        # --- 2. Initial Condition Loss ---
        U_ic = model.apply(params, batch['ic'], train=True)        
        loss_ic = jnp.mean(U_ic[..., 0]**2) + jnp.mean(U_ic[..., 1]**2 + U_ic[..., 2]**2)

        # --- 3. Boundary Conditions ---
        
        # A. Inflow Boundary (Flux prescribed)
        t_inflow = batch['bc_inflow'][..., 2]
        Q_target = bc_fn_static(t_inflow) # m^3/s
        flux_target_x = Q_target / 100.0  # m^2/s (assuming 100m width approx, or derived from edge)

        loss_inflow_x = loss_boundary_dirichlet_hu(model, params, batch['bc_inflow'], flux_target_x)
        loss_inflow_y = loss_boundary_dirichlet_hv(model, params, batch['bc_inflow'], jnp.zeros_like(flux_target_x))
        loss_bc_inflow = loss_inflow_x + loss_inflow_y

        # B. Wall Boundaries (Generalized Slip)
        loss_bc_wall = loss_slip_wall_generalized(model, params, batch['bc_wall'])
        
        total_bc = loss_bc_inflow + loss_bc_wall

        # Data Loss
        data_batch_data = batch.get('data', jnp.empty((0,6), dtype=DTYPE))
        if not data_free and 'data' in active_loss_keys_base and data_batch_data.shape[0] > 0:
             loss_data = compute_data_loss(model, params, data_batch_data, config)
        else:
             loss_data = 0.0

        terms = {
            'pde': loss_pde,
            'neg_h': loss_neg_h,
            'ic': loss_ic,
            'bc': total_bc, 
            'data': loss_data
        }

        terms_with_defaults = {k: terms.get(k, 0.0) for k in weights_dict.keys()}
        total = total_loss(terms_with_defaults, weights_dict)
        return total, terms

    (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt_state = optimiser.update(grads, opt_state, params, value=loss_val)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, metrics, loss_val

# JIT Compile
train_step_jitted = jax.jit(train_step, static_argnames=['model', 'optimiser', 'config', 'bc_fn_static', 'weights_dict', 'data_free'])

def main(config_path: str):
    """
    Main training loop for Benchmark Test 5 Scenario.
    """
    #--- 1. LOAD CONFIGURATION (MUTABLE) ---
    cfg_dict = load_config(config_path)
    
    print("Info: Running Benchmark Test 5 Scenario model training...")

    # --- 2. SETUP DATA & COMPUTE DOMAIN EXTENT ---
    scenario_name = cfg_dict.get('scenario')
    if not scenario_name:
         print(f"Error: 'scenario' key must be set in config '{config_path}'.")
         sys.exit(1)
         
    base_data_path = os.path.join("data", scenario_name)

    # A. Init Irregular Domain Sampler & Calculate lx, ly
    # Check for 'domain.npz' (user preference) or 'domain_artifacts.npz' (legacy)
    artifacts_path = os.path.join(base_data_path, "domain.npz")
    if not os.path.exists(artifacts_path):
        artifacts_path = os.path.join(base_data_path, "domain_artifacts.npz")
        
    if not os.path.exists(artifacts_path):
        print(f"Error: Domain artifacts file not found (checked 'domain.npz' and 'domain_artifacts.npz').")
        sys.exit(1)
    
    print(f"Loading domain geometry from: {artifacts_path}")
    domain_sampler = IrregularDomainSampler(artifacts_path)

    # --- CALCULATE DOMAIN EXTENT ---
    # tri_coords shape: (N_tri, 3, 2)
    all_coords = domain_sampler.tri_coords.reshape(-1, 2)
    min_vals = jnp.min(all_coords, axis=0)
    max_vals = jnp.max(all_coords, axis=0)
    
    x_min, y_min = float(min_vals[0]), float(min_vals[1])
    x_max, y_max = float(max_vals[0]), float(max_vals[1])
    
    calc_lx = x_max - x_min
    calc_ly = y_max - y_min
    
    print(f"Computed Domain Extent:")
    print(f"  X Range: [{x_min:.4f}, {x_max:.4f}]")
    print(f"  Y Range: [{y_min:.4f}, {y_max:.4f}]")
    print(f"  Calculated Dimensions: lx = {calc_lx:.4f}, ly = {calc_ly:.4f}")
    
    # Update Config with calculated values
    if 'domain' not in cfg_dict: cfg_dict['domain'] = {}
    cfg_dict['domain']['lx'] = calc_lx
    cfg_dict['domain']['ly'] = calc_ly
    cfg_dict['domain']['x_min'] = x_min
    cfg_dict['domain']['x_max'] = x_max
    cfg_dict['domain']['y_min'] = y_min
    cfg_dict['domain']['y_max'] = y_max

    h_scale = 6.0  
    hu_scale = 2.0 
    hv_scale = 2.0

    if 'model' not in cfg_dict: cfg_dict['model'] = {}
    # This list corresponds to the output channels [h, hu, hv]
    cfg_dict['model']['output_scales'] = (h_scale, hu_scale, hv_scale)
    
    print(f"Active Output Scaling: {cfg_dict['model']['output_scales']}")

    # --- 3. FINALIZE CONFIG & INIT MODEL ---
    # Now that config has correct dimensions, freeze it and init model
    cfg = FrozenDict(cfg_dict)

    try:
        models_module = importlib.import_module("src.models")
        model_class = getattr(models_module, cfg["model"]["name"])
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not find model class '{cfg['model']['name']}' in src/models.py") from e
    
    key = random.PRNGKey(cfg["training"]["seed"])
    model_key, train_key, val_key = random.split(key, 3)
    model, params = init_model(model_class, model_key, cfg)

    # --- 4. Setup Directories ---
    config_base = os.path.splitext(os.path.basename(cfg['CONFIG_PATH']))[0]
    trial_name = generate_trial_name(config_base)
    results_dir = os.path.join("results", trial_name)
    model_dir = os.path.join("models", trial_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --- 5. Prepare Loss Weights ---
    static_weights_dict = {k.replace('_weight',''):v for k,v in cfg["loss_weights"].items()}
    active_loss_term_keys = [k for k, v in static_weights_dict.items() if v > 0]
    current_weights_dict = FrozenDict({k: static_weights_dict[k] for k in active_loss_term_keys})

    # --- 6. Load Remaining Assets ---

    # B. Load Bathymetry (REQUIRED)
    dem_path = os.path.join(base_data_path, "Test5DEM.asc")
    if not os.path.exists(dem_path):
        print(f"Error: DEM file not found at {dem_path}")
        sys.exit(1)
    print(f"Loading Bathymetry from {dem_path}...")
    load_bathymetry(dem_path)
    
    # C. Load Boundary Condition Function
    bc_csv_path = os.path.join(base_data_path, "Test5BC.csv")
    if not os.path.exists(bc_csv_path):
        # Fallback to Test4BC if shared
        bc_csv_path_alt = os.path.join(base_data_path, "Test4BC.csv")
        if os.path.exists(bc_csv_path_alt):
             bc_csv_path = bc_csv_path_alt
        else:
            print(f"Error: Boundary condition CSV file not found at {bc_csv_path}.")
            sys.exit(1)
    bc_fn_static = load_boundary_condition(bc_csv_path)

    # D. Load Validation and Training Data
    val_points, h_true_val = None, None
    data_points_full = None
    
    data_free_flag = cfg.get("data_free")
    
    if data_free_flag is False:
        print("Info: 'data_free: false' found in config. Activating data-driven mode.")
        has_data_loss = True
        data_free = False
    else:
        if data_free_flag is None:
            print("Warning: 'data_free' flag not specified in config. Defaulting to 'data_free: true'.")
        else:
            print("Info: 'data_free: true' found in config. Data loss term will be disabled.")
        has_data_loss = False
        data_free = True

    training_data_file = os.path.join(base_data_path, "training_dataset_sample.npy")
    if has_data_loss: 
        if os.path.exists(training_data_file):
            try:
                print(f"Loading TRAINING data from: {training_data_file}")
                data_points_full = jnp.load(training_data_file).astype(DTYPE) 
                if data_points_full.shape[0] == 0:
                     print("Warning: Training data file is empty. Disabling data loss.")
                     data_points_full = None
                     has_data_loss = False
                else:
                     data_weight = static_weights_dict.get('data', 0.0)
                     print(f"Using {data_points_full.shape[0]} points for data loss term (weight={data_weight:.2e}).")
            except Exception as e:
                print(f"Error loading training data file {training_data_file}: {e}")
                data_points_full = None
                has_data_loss = False
        else:
            print(f"Warning: Training data file not found at {training_data_file}.")
            has_data_loss = False
    
    data_free = not has_data_loss 

    # E. Load Validation Data (Optional)
    validation_data_file = os.path.join(base_data_path, "validation_gauges_ground_truth.npy")
    validation_data_loaded = False
    full_val_data = None
    
    if os.path.exists(validation_data_file):
        try:
            print(f"Loading VALIDATION data from: {validation_data_file}")
            loaded_val_data = jnp.load(validation_data_file).astype(DTYPE)
            full_val_data = loaded_val_data 
            val_points = loaded_val_data[:, [1, 2, 0]]
            h_true_val = loaded_val_data[:, 3]
            num_val_points = val_points.shape[0]
            if num_val_points > 0:
                validation_data_loaded = True
                val_points_all = val_points 
                h_true_val_all = h_true_val
            else:
                 print("Warning: No validation points remaining after masking.")
        except Exception as e:
            print(f"Error loading validation data: {e}")
            val_points, h_true_val = None, None
    else:
        print(f"Warning: Validation data not found at {validation_data_file}.")

    # --- 7. Initialize Aim & Log Source Code ---
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

        artifact_storage_path = os.path.join(aim_repo_path, "aim_artifacts")
        os.makedirs(artifact_storage_path, exist_ok=True)
        abs_artifact_path = os.path.abspath(artifact_storage_path)
        aim_run.set_artifacts_uri(f"file://{abs_artifact_path}")
        print(f"Set Aim artifact storage to: {abs_artifact_path}")
        
        # Log basics
        hparams_to_log = copy.deepcopy(cfg_dict)
        aim_run["hparams"] = hparams_to_log
        aim_run['flags'] = {"scenario_type": "experiment_7"}

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

    # --- 8. Summary ---
    print(f"\n--- Training Started: {trial_name} ---")
    print(f"Model: {cfg['model']['name']}, Epochs: {cfg['training']['epochs']}")
    print(f"Data Loss Active: {has_data_loss} (Final Data-Free: {data_free})")
    print(f"Active Loss Terms: {active_loss_term_keys}")
    print(f"Initial Weights: {current_weights_dict}")

    start_time = time.time()
    global_step = 0

    # --- 9. Data Generation Setup ---
    sampling_cfg = cfg["sampling"]
    batch_size = cfg["training"]["batch_size"]
    domain_cfg = cfg["domain"]
    
    n_pde = get_sample_count(sampling_cfg, "n_points_pde", 1000)
    n_ic = get_sample_count(sampling_cfg, "n_points_ic", 100)
    n_bc_inflow = get_sample_count(sampling_cfg, "n_points_bc_inflow", 100)
    n_bc_wall = get_sample_count(sampling_cfg, "n_points_bc_domain", 100)

    bc_counts = [n_pde//batch_size, n_ic//batch_size, n_bc_wall//batch_size, n_bc_inflow//batch_size]
    if not data_free and data_points_full is not None:
        bc_counts.append(data_points_full.shape[0] // batch_size)

    num_batches = max(bc_counts) if bc_counts else 0
    if num_batches == 0:
        print(f"Error: Batch size {batch_size} is too large for sample counts.")
        return -1.0
    print(f"Batches per epoch: {num_batches}")

    # --- Optimizer ---
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
            accumulation_size=num_batches*int(reduce_on_plateau_cfg.get("accumulation_factor", 1)),
            min_scale=float(reduce_on_plateau_cfg.get("min_scale", 1e-6)),
        ),
    )
    opt_state = optimiser.init(params)

    def generate_epoch_data(key):
        k1, k2, k3, k4, k5 = random.split(key, 5)
        
        # Interior
        pde_pts = domain_sampler.sample_interior(k1, n_pde, (0., domain_cfg["t_final"]))
        pde_data = get_batches_tensor(k1, pde_pts, batch_size, num_batches)
        
        # IC
        ic_pts = domain_sampler.sample_interior(k2, n_ic, (0., 0.))
        ic_data = get_batches_tensor(k2, ic_pts, batch_size, num_batches)
        
        # BCs
        bc_inflow_pts = domain_sampler.sample_boundary(k3, n_bc_inflow, (0., domain_cfg["t_final"]), 'inflow')
        bc_inflow = get_batches_tensor(k3, bc_inflow_pts, batch_size, num_batches)

        bc_wall_pts = domain_sampler.sample_boundary(k4, n_bc_wall, (0., domain_cfg["t_final"]), 'wall')
        bc_wall = get_batches_tensor(k4, bc_wall_pts, batch_size, num_batches)
        
        # Data - Now actually uses the data
        if not data_free and data_points_full is not None:
             data_d = get_batches_tensor(k5, data_points_full, batch_size, num_batches)
        else:
             data_d = jnp.zeros((num_batches, 0, 6), dtype=DTYPE)

        return {
            'pde': pde_data, 'ic': ic_data, 
            'bc_inflow': bc_inflow, 'bc_wall': bc_wall, 
            'data': data_d
        }
    
    generate_epoch_data_jitted = jax.jit(generate_epoch_data)

    # Scan Body
    def scan_body(carry, batch_data):
        curr_params, curr_opt_state = carry
        current_all_batches = {
            'pde': batch_data['pde'],
            'ic': batch_data['ic'],
            'bc_inflow': batch_data['bc_inflow'],
            'bc_wall': batch_data['bc_wall'], # Generic wall
            'data': batch_data['data']
        }
        new_params, new_opt_state, terms, total = train_step_jitted(
            model, optimiser, curr_params, curr_opt_state,
            current_all_batches, cfg, data_free, bc_fn_static, current_weights_dict
        )
        return (new_params, new_opt_state), (terms, total)

    # --- 10. Training Loop ---
    best_nse_stats = {
        'nse': -jnp.inf, 'rmse': jnp.inf, 'epoch': 0, 'global_step': 0,
        'time_elapsed_seconds': 0.0, 'total_weighted_loss': 0.0, 'unweighted_losses': {}
    }
    
    best_loss_stats = {
        'total_weighted_loss': jnp.inf, 'epoch': 0, 'global_step': 0,
        'time_elapsed_seconds': 0.0, 'nse': -jnp.inf, 'rmse': jnp.inf, 'unweighted_losses': {}
    }
    
    best_params_nse = None
    best_params_loss = None 

    try:
        for epoch in range(cfg["training"]["epochs"]):
            epoch_start_time = time.time()
            train_key, epoch_key = random.split(train_key)
            scan_inputs = generate_epoch_data_jitted(epoch_key)
            
            (params, opt_state), (batch_losses_unweighted_stacked, batch_total_weighted_loss_stacked) = lax.scan(
                scan_body, (params, opt_state), scan_inputs
            )
            
            global_step += num_batches
            
            # --- Aggregate Losses ---
            epoch_losses_unweighted_sum = {k: jnp.sum(v) for k, v in batch_losses_unweighted_stacked.items()}
            epoch_total_weighted_loss_sum = jnp.sum(batch_total_weighted_loss_stacked)

            avg_losses_unweighted = {k: float(v) / num_batches for k, v in epoch_losses_unweighted_sum.items()}
            avg_total_weighted_loss = float(epoch_total_weighted_loss_sum) / num_batches

            # --- LR Extraction ---
            current_lr = cfg["training"]["learning_rate"]
            current_scale = 1.0
            base_lr_val = cfg["training"]["learning_rate"]
            try:
                if hasattr(opt_state[-1], 'scale'):
                    current_scale = float(opt_state[-1].scale)
                    current_lr = base_lr_val * current_scale
            except Exception as e:
                if epoch == 0: 
                    print(f"Warning: Failed to extract LR scale: {e}")            

            # Validation
            nse_val, rmse_val = -jnp.inf, jnp.inf
            if validation_data_loaded:
                try:
                    U_val = model.apply(params, val_points_all, train=False)
                    nse_val = nse(h_true_val_all, U_val[..., 0])
                    rmse_val = rmse(h_true_val_all, U_val[..., 0])
                except: pass

            # --- Update Best Model Statistics ---
            if nse_val > best_nse_stats['nse']:
                best_nse_stats.update({
                    'nse': nse_val, 'rmse': rmse_val, 'epoch': epoch, 'global_step': global_step,
                    'time_elapsed_seconds': time.time() - start_time,
                    'total_weighted_loss': avg_total_weighted_loss,
                    'unweighted_losses': {k: float(v) for k, v in avg_losses_unweighted.items()}
                })
                best_params_nse = copy.deepcopy(params)
                if nse_val > -jnp.inf:
                    print(f"    ---> New best NSE: {best_nse_stats['nse']:.6f} at epoch {epoch+1}")
            
            if avg_total_weighted_loss < best_loss_stats['total_weighted_loss']:
                best_loss_stats.update({
                    'total_weighted_loss': avg_total_weighted_loss, 'epoch': epoch, 'global_step': global_step,
                    'time_elapsed_seconds': time.time() - start_time,
                    'nse': nse_val, 'rmse': rmse_val,
                    'unweighted_losses': {k: float(v) for k, v in avg_losses_unweighted.items()}
                })
                best_params_loss = copy.deepcopy(params) 

            # Reporting
            freq = cfg.get("reporting", {}).get("epoch_freq", 100)
            epoch_time = time.time() - epoch_start_time
            if (epoch + 1) % freq == 0:
                print_epoch_stats(
                    epoch, global_step, start_time, avg_total_weighted_loss,
                    avg_losses_unweighted.get('pde', 0.0), 
                    avg_losses_unweighted.get('ic', 0.0), 
                    avg_losses_unweighted.get('bc', 0.0),
                    0.0, 
                    avg_losses_unweighted.get('data', 0.0),
                    avg_losses_unweighted.get('neg_h', 0.0),
                    nse_val, rmse_val, epoch_time
                )
                print(f"    LR Status: LR={current_lr:.2e}, Base={base_lr_val:.2e}, Scale={current_scale:.2e}")

            if aim_run:
                epoch_metrics_to_log = {
                    'validation_metrics': {'nse': nse_val, 'rmse': rmse_val},
                    'epoch_avg_losses': avg_losses_unweighted,
                    'epoch_avg_total_weighted_loss': avg_total_weighted_loss,
                    'optimization': {'total_loss': avg_total_weighted_loss}, 
                    'system_metrics': {'epoch_time': epoch_time},
                    'training_metrics': {'learning_rate': float(current_lr)}
                }
                log_metrics(aim_run, step=global_step, epoch=epoch, metrics=epoch_metrics_to_log)

            # --- Early Stopping Check ---
            min_epochs = cfg.get("device", {}).get("early_stop_min_epochs", float('inf'))
            patience = cfg.get("device", {}).get("early_stop_patience", float('inf'))

            if epoch >= min_epochs and (epoch - best_nse_stats['epoch']) >= patience:
                print(f"--- Early stopping triggered at epoch {epoch+1} ---")
                print(f"Best NSE {best_nse_stats['nse']:.6f} achieved at epoch {best_nse_stats['epoch']+1}.")
                break

    except KeyboardInterrupt:
        print("\n--- Training interrupted ---")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    # --- 11. Post-Training (Save & Plot) ---
    finally:
        total_time = time.time() - start_time
        print_final_summary(total_time, best_nse_stats, best_loss_stats)

        final_params = best_params_loss if best_params_loss is not None else best_params_nse

        if aim_run:
            try:
                summary_best_nse = best_nse_stats.copy()
                summary_best_nse['epoch'] = best_nse_stats.get('epoch', 0) + 1
                summary_best_loss = best_loss_stats.copy()
                summary_best_loss['epoch'] = best_loss_stats.get('epoch', 0) + 1
                
                summary_metrics = {
                    'best_validation_model': summary_best_nse,
                    'best_loss_model': summary_best_loss,
                    'final_system': {
                        'total_training_time_seconds': total_time,
                        'total_epochs_run': (epoch + 1) if 'epoch' in locals() else 0,
                        'total_steps_run': global_step
                    }
                }
                aim_run['summary'] = summary_metrics
                print("Summary metrics logged to Aim.")
            except Exception as e:
                 print(f"Warning: Error logging summary metrics to Aim: {e}")   

        if ask_for_confirmation():
            if final_params is not None:
                saved_model_path = save_model(final_params, model_dir, trial_name)

                if aim_run and saved_model_path:
                    try:
                        aim_run.log_artifact(saved_model_path, name='model_weights.pkl')
                        print(f"Logged model artifact to Aim.")
                    except Exception as e_mod:
                        print(f"Warning: Failed to log model artifact: {e_mod}")

                print("Generating Test 5 plots...")
                t_plot = jnp.arange(0., cfg['domain']['t_final'], 60.0, dtype=DTYPE)
                
                output_csv_path = os.path.join(base_data_path, "Test5output.csv")
                output_points = []
                if os.path.exists(output_csv_path):
                    try:
                        df_out = pd.read_csv(output_csv_path)
                        if 'X' in df_out.columns and 'Y' in df_out.columns:
                            for idx, row in df_out.iterrows():
                                output_points.append((row['X'], row['Y'], f"Point_{idx+1}"))
                        else:
                            arr_out = df_out.values
                            for i in range(arr_out.shape[0]):
                                output_points.append((arr_out[i, 0], arr_out[i, 1], f"Point_{i+1}"))
                        print(f"Loaded {len(output_points)} output points from CSV.")
                    except Exception as e:
                        print(f"Warning: Could not read Test5output.csv: {e}")
                
                if not output_points:
                    print("Using default representative points (Center).")
                    cx, cy = (x_max+x_min)/2, (y_max+y_min)/2
                    output_points = [(cx, cy, "Center_Point")]

                def plot_gauge(x, y, name, filename):
                    pts = jnp.stack([jnp.full_like(t_plot, x), jnp.full_like(t_plot, y), t_plot], axis=-1)
                    U = model.apply(final_params, pts, train=False)
                    h_pred = U[..., 0]
                    
                    plt.figure(figsize=(10, 6))

                    if full_val_data is not None:
                        val_np = np.array(full_val_data)
                        mask = np.isclose(val_np[:, 1], x) & np.isclose(val_np[:, 2], y)
                        gauge_data = val_np[mask]
                        
                        if gauge_data.shape[0] > 0:
                            gauge_data = gauge_data[gauge_data[:, 0].argsort()]
                            plt.plot(gauge_data[:, 0], gauge_data[:, 3], 'k--', linewidth=1.5, alpha=0.7, label=f'Baseline {name}')

                    plt.plot(t_plot, h_pred, label=f'Predicted h @ ({x:.1f},{y:.1f})')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Water Level h (m)')
                    plt.title(f'{name} - Water Level vs Time')
                    plt.legend()
                    plt.grid(True)
                    path = os.path.join(results_dir, filename)
                    plt.savefig(path)
                    plt.close()
                    if aim_run:
                        aim_run.track(Image(path), name=filename)

                for px, py, pname in output_points:
                     plot_gauge(px, py, pname, f"{pname}_timeseries.png")
                     
                print(f"Plots saved to {results_dir}")
            else:
                print("No model parameters found to save.")

        else:
            print("Save aborted by user. Deleting artifacts...")
            try:
                if aim_run and run_hash and aim_repo:
                    aim_repo.delete_run(run_hash)
                    print("Aim run deleted.")
                
                if os.path.exists(results_dir):
                    shutil.rmtree(results_dir)
                    print(f"Deleted results directory: {results_dir}")
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                    print(f"Deleted model directory: {model_dir}")

                if run_hash:
                    run_artifact_dir = os.path.join("aim_repo", "aim_artifacts", run_hash)
                    if os.path.exists(run_artifact_dir):
                        shutil.rmtree(run_artifact_dir)
                        print(f"Deleted run artifact directory: {run_artifact_dir}")

                print("Cleanup complete.")
            except Exception as e:
                print(f"Error during cleanup: {e}")

        if aim_run:
            try:
                aim_run.close()
                print("Aim run closed.")
            except Exception as e:
                 print(f"Warning: Error closing Aim run: {e}")

    return best_nse_stats['nse']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified PINN training script for SWE (Test 5 - Irregular).")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path: sys.path.insert(0, project_root)

    main(args.config)