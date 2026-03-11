"""Experiment 4 — Terrain slope in x+y directions (Phase 2).

Extends terrain slope to both x and y directions.
Requires: configs/experiment_4.yaml, data/experiment_4/
Builds on: Experiment 3.
"""
import os
import sys
import time
import copy
import argparse
import importlib
import itertools
from typing import Any, Dict, Tuple
import shutil
import pandas as pd # Added for reading output CSV

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
    sample_domain, 
    get_batches_tensor,
    get_sample_count,
    bathymetry_fn,
    load_boundary_condition,
    load_bathymetry,
    sample_lhs,
    load_validation_data,
)
from src.models import init_model
from src.losses import (
    compute_pde_loss,
    loss_boundary_dirichlet_hu,
    loss_boundary_dirichlet_hv,
    loss_boundary_wall_horizontal,
    loss_boundary_wall_vertical,
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
    """
    Performs one step of gradient descent for Benchmark Test 2.
    """
    
    active_loss_keys_base = list(weights_dict.keys())

    def loss_fn(params):
        
        terms = {}
        # --- 1. PDE Loss (Physics + Bathymetry) ---
        loss_pde = compute_pde_loss(model, params, batch['pde'], config)
        loss_neg_h = compute_neg_h_loss(model, params, batch['pde'])
        
        # --- 2. Initial Condition Loss (Dry Bed: h=0, u=0, v=0) ---
        # Test 2 Spec: "Initial condition: Dry bed."
        U_ic = model.apply(params, batch['ic'], train=True)        
        h_ic_pred = U_ic[..., 0]
        hu_ic_pred = U_ic[..., 1]
        hv_ic_pred = U_ic[..., 2]

        # Target is strictly zero
        loss_ic_h = jnp.mean(h_ic_pred**2)
        loss_ic_vel = jnp.mean(hu_ic_pred**2 + hv_ic_pred**2) 
        loss_ic = loss_ic_h + loss_ic_vel

        # --- 3. Boundary Conditions ---
        
        # A. Inflow Boundary (Part of Left Edge): Flux BC
        # Test 2 Spec: Inflow hydrograph along 100m line from NW corner.
        # We model this as imposing flux q = Q(t) / Width. Width = 100m.
        
        # Get Time and interpolated Flow Q(t)
        t_inflow = batch['bc_inflow'][..., 2]
        Q_target_x = bc_fn_static(t_inflow) # Returns m^3/s
        flux_target_x = Q_target_x / 100.0    # Discharge per unit width (m^2/s)

        loss_inflow_x=loss_boundary_dirichlet_hu(model, params, batch['bc_inflow'], flux_target_x)
        loss_inflow_y=loss_boundary_dirichlet_hv(model, params, batch['bc_inflow'], jnp.zeros_like(flux_target_x))
        
        loss_bc_inflow = loss_inflow_x + loss_inflow_y

        # B. Left Wall (Remaining part of Left Edge): Slip Wall
        loss_bc_left_wall = loss_boundary_wall_vertical(model, params, batch['bc_left_wall'])
        
        # C. Right Boundary: Slip Walls (No flux x)
        loss_bc_right = loss_boundary_wall_vertical(model, params, batch['bc_right'])
        
        # D. Top & Bottom Boundaries: Slip Walls (No flux y)
        loss_bc_top = loss_boundary_wall_horizontal(model, params, batch['bc_top'])
        loss_bc_bottom = loss_boundary_wall_horizontal(model, params, batch['bc_bottom'])
        
        # Aggregate BC losses
        total_bc = loss_bc_inflow + loss_bc_left_wall + loss_bc_right + loss_bc_top + loss_bc_bottom

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
            'data': loss_data if not data_free and 'data' in active_loss_keys_base and data_batch_data.shape[0] > 0 else 0.0
        }

        # --- 4. Weighted Sum ---
        terms_with_defaults = {k: terms.get(k, 0.0) for k in weights_dict.keys()}
        total = total_loss(terms_with_defaults, weights_dict)
        
        return total, terms

    # Calculate Gradients
    (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Update Parameters
    updates, new_opt_state = optimiser.update(grads, opt_state, params, value=loss_val)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, metrics, loss_val

# JIT Compile
train_step_jitted = jax.jit(train_step, static_argnames=['model', 'optimiser', 'config', 'bc_fn_static', 'weights_dict', 'data_free'])

def main(config_path: str):
    """
    Main training loop for Benchmark Test 2 Scenario.
    """
    
    #--- 1. LOAD CONFIGURATION ---
    cfg_dict = load_config(config_path)
    cfg = FrozenDict(cfg_dict)

    print("Info: Running Benchmark Test 2 Scenario model training...")

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

    # --- 4. Prepare Loss Weights ---
    static_weights_dict = {k.replace('_weight',''):v for k,v in cfg["loss_weights"].items()}
    active_loss_term_keys = [k for k, v in static_weights_dict.items() if v > 0]
    current_weights_dict = FrozenDict({k: static_weights_dict[k] for k in active_loss_term_keys})

    # --- 5. Load Data Assets ---
    scenario_name = cfg.get('scenario')
    if not scenario_name:
         print(f"Error: 'scenario' key must be set in config '{config_path}'.")
         sys.exit(1)
         
    base_data_path = os.path.join("data", scenario_name)
    
    # A. Load Bathymetry (Test 2 DEM)
    dem_path = os.path.join(base_data_path, "Test2DEM.asc")
    if not os.path.exists(dem_path):
        print(f"Error: DEM file not found at {dem_path}")
        sys.exit(1)
    print(f"Loading Bathymetry from {dem_path}...")
    load_bathymetry(dem_path)

    # B. Load Boundary Condition Function
    bc_csv_path = os.path.join(base_data_path, "Test2_BC.csv")
    if not os.path.exists(bc_csv_path):
        print(f"Error: Boundary condition CSV file not found at {bc_csv_path}.")
        sys.exit(1)
    bc_fn_static = load_boundary_condition(bc_csv_path)

    # --- 5b. Load Validation and Training Data ---
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

    # C. Load Validation Data (Optional)
    validation_data_file = os.path.join(base_data_path, "validation_gauges_ground_truth.npy")
    validation_data_loaded = False
    full_val_data = None
    
    if os.path.exists(validation_data_file):
        try:
            print(f"Loading VALIDATION data from: {validation_data_file}")
            full_val_data, val_points, val_targets = load_validation_data(validation_data_file, dtype=DTYPE)
            h_true_val = val_targets[:, 0]
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

        artifact_storage_path = os.path.join(aim_repo_path, "aim_artifacts")
        os.makedirs(artifact_storage_path, exist_ok=True)
        abs_artifact_path = os.path.abspath(artifact_storage_path)
        aim_run.set_artifacts_uri(f"file://{abs_artifact_path}")
        print(f"Set Aim artifact storage to: {abs_artifact_path}")
        
        # Log basics
        hparams_to_log = copy.deepcopy(cfg_dict)
        aim_run["hparams"] = hparams_to_log
        aim_run['flags'] = {"scenario_type": "experiment_4"}

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
    print(f"Data Loss Active: {has_data_loss} (Final Data-Free: {data_free})")
    print(f"Active Loss Terms: {active_loss_term_keys}")
    print(f"Initial Weights: {current_weights_dict}")

    start_time = time.time()
    global_step = 0

    # --- 8. Data Generation Setup ---
    sampling_cfg = cfg["sampling"]
    batch_size = cfg["training"]["batch_size"]
    domain_cfg = cfg["domain"]
    
    n_pde = get_sample_count(sampling_cfg, "n_points_pde", 1000)
    n_ic = get_sample_count(sampling_cfg, "n_points_ic", 100)
    n_bc_domain_wall = get_sample_count(sampling_cfg, "n_points_bc_domain", 100)
    n_bc_inflow = get_sample_count(sampling_cfg, "n_points_bc_inflow", 100)
    n_bc_per_wall = max(5, n_bc_domain_wall // 4)

    bc_counts = [n_pde//batch_size, n_ic//batch_size, n_bc_per_wall//batch_size, n_bc_inflow//batch_size]
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

    # JIT Data Generator
    def generate_epoch_data(key):
        key, pde_key, ic_key, bc_keys, data_key = random.split(key, 5)
        
        # PDE: Domain is 2000x2000, t_final from config (usually 48h)
        if n_pde // batch_size > 0:
            pde_pts = sample_lhs(pde_key, n_pde, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"]))
            pde_data = get_batches_tensor(pde_key, pde_pts, batch_size, num_batches)
        else:
            pde_data = jnp.zeros((num_batches, 0, 3), dtype=DTYPE)

        # IC: Dry Bed everywhere
        if n_ic // batch_size > 0:
            ic_pts = sample_lhs(ic_key, n_ic, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., 0.))
            ic_data = get_batches_tensor(ic_key, ic_pts, batch_size, num_batches)
        else:
            ic_data = jnp.zeros((num_batches, 0, 3), dtype=DTYPE)
            
        # BCs
        # Test 2 Logic: 
        # Left Boundary (x=0) is split:
        #   - Inflow: y in [1900, 2000] (North-West 100m)
        #   - Wall: y in [0, 1900]
        # Right, Top, Bottom are simple walls.
        
        l_in_key, l_wall_key, r_key, b_key, t_key = random.split(bc_keys, 5)
        
        def get_bc_batch(k, n, x_range, y_range):
            if n // batch_size > 0:
                pts = sample_lhs(k, n, x_range, y_range, (0., domain_cfg["t_final"]))
                return get_batches_tensor(k, pts, batch_size, num_batches)
            return jnp.zeros((num_batches, 0, 3), dtype=DTYPE)

        # Split Left Boundary
        y_inflow_start = domain_cfg["ly"] - 100.0 # e.g., 1900m
        bc_inflow = get_bc_batch(l_in_key, n_bc_inflow, (0., 0.), (y_inflow_start, domain_cfg["ly"]))
        bc_left_wall = get_bc_batch(l_wall_key, n_bc_per_wall, (0., 0.), (0., y_inflow_start))

        # Other Boundaries
        bc_right = get_bc_batch(r_key, n_bc_per_wall, (domain_cfg["lx"], domain_cfg["lx"]), (0., domain_cfg["ly"]))
        bc_bot = get_bc_batch(b_key, n_bc_per_wall, (0., domain_cfg["lx"]), (0., 0.))
        bc_top = get_bc_batch(t_key, n_bc_per_wall, (0., domain_cfg["lx"]), (domain_cfg["ly"], domain_cfg["ly"]))

        # Data
        data_data = jnp.zeros((num_batches, 0, 6), dtype=DTYPE)
        if not data_free and data_points_full is not None:
             data_data = get_batches_tensor(data_key, data_points_full, batch_size, num_batches)

        return {
            'pde': pde_data,
            'ic': ic_data,
            'bc': {
                'inflow': bc_inflow, 
                'left_wall': bc_left_wall, 
                'right': bc_right, 
                'bottom': bc_bot, 
                'top': bc_top
            },
            'data': data_data
        }
    
    generate_epoch_data_jitted = jax.jit(generate_epoch_data)

    # Scan Body
    def scan_body(carry, batch_data):
        curr_params, curr_opt_state = carry
        
        current_all_batches = {
            'pde': batch_data['pde'],
            'ic': batch_data['ic'],
            'bc_inflow': batch_data['bc']['inflow'],
            'bc_left_wall': batch_data['bc']['left_wall'],
            'bc_right': batch_data['bc']['right'],
            'bc_bottom': batch_data['bc']['bottom'],
            'bc_top': batch_data['bc']['top'],
            'data': batch_data['data']
        }

        new_params, new_opt_state, terms, total = train_step_jitted(
            model, optimiser, curr_params, curr_opt_state,
            current_all_batches, cfg, data_free, bc_fn_static, current_weights_dict
        )
        return (new_params, new_opt_state), (terms, total)

    # --- 9. Training Loop ---
    best_nse_stats = {
        'nse': -jnp.inf, 'rmse': jnp.inf, 'epoch': 0, 'global_step': 0,
        'time_elapsed_seconds': 0.0, 'total_weighted_loss': 0.0, 'unweighted_losses': {}
    }
    
    best_loss_stats = {
        'total_weighted_loss': jnp.inf, 'epoch': 0, 'global_step': 0,
        'time_elapsed_seconds': 0.0, 'nse': -jnp.inf, 'rmse': jnp.inf, 'unweighted_losses': {}
    }
    
    best_params_nse = None
    best_params_loss = None # Added tracking for best loss model

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
            # Sum over batches dimension
            epoch_losses_unweighted_sum = {k: jnp.sum(v) for k, v in batch_losses_unweighted_stacked.items()}
            epoch_total_weighted_loss_sum = jnp.sum(batch_total_weighted_loss_stacked)

            avg_losses_unweighted = {k: float(v) / num_batches for k, v in epoch_losses_unweighted_sum.items()}
            avg_total_weighted_loss = float(epoch_total_weighted_loss_sum) / num_batches

            # --- LR Extraction ---
            # Extract LR status from opt_state (chain: clip, adam, reduce_on_plateau)
            current_lr = cfg["training"]["learning_rate"]
            current_scale = 1.0
            base_lr_val = cfg["training"]["learning_rate"]
            try:
                # Access the state of the last transformation (reduce_on_plateau)
                # opt_state is a tuple corresponding to the chain elements
                # Ensure we cast the JAX array to a standard float
                if hasattr(opt_state[-1], 'scale'):
                    current_scale = float(opt_state[-1].scale)
                    current_lr = base_lr_val * current_scale
            except Exception as e:
                # Removing silent failure to aid debugging
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
                best_params_loss = copy.deepcopy(params) # Update best params for loss

            # Reporting
            freq = cfg.get("reporting", {}).get("epoch_freq", 100)
            epoch_time = time.time() - epoch_start_time
            if (epoch + 1) % freq == 0:
                print_epoch_stats(
                    epoch, global_step, start_time, avg_total_weighted_loss,
                    avg_losses_unweighted.get('pde', 0.0), 
                    avg_losses_unweighted.get('ic', 0.0), 
                    avg_losses_unweighted.get('bc', 0.0),
                    0.0, # Placeholder for unused loss term to satisfy signature
                    avg_losses_unweighted.get('data', 0.0),
                    avg_losses_unweighted.get('neg_h', 0.0),
                    nse_val, rmse_val, epoch_time
                )
                # MOVED: Printing LR status here
                print(f"    LR Status: LR={current_lr:.2e}, Base={base_lr_val:.2e}, Scale={current_scale:.2e}")

            if aim_run:
                epoch_metrics_to_log = {
                    'validation_metrics': {'nse': nse_val, 'rmse': rmse_val},
                    'epoch_avg_losses': avg_losses_unweighted,
                    'epoch_avg_total_weighted_loss': avg_total_weighted_loss,
                    'optimization': {'total_loss': avg_total_weighted_loss}, # Added total_loss explicitly
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

    # --- 10. Post-Training (Save & Plot) ---
    finally:
        total_time = time.time() - start_time
        print_final_summary(total_time, best_nse_stats, best_loss_stats)

        # Decide which params to save (User preference: Best Loss Model)
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

                # --- NEW: Log Model as Artifact to Aim ---
                if aim_run and saved_model_path:
                    try:
                        aim_run.log_artifact(saved_model_path, name='model_weights.pkl')
                        print(f"Logged model artifact to Aim.")
                    except Exception as e_mod:
                        print(f"Warning: Failed to log model artifact: {e_mod}")

                # --- Plotting Specific to Test 2 ---
                
                print("Generating Test 2 plots...")
                t_plot = jnp.arange(0., cfg['domain']['t_final'], 60.0, dtype=DTYPE)
                
                # Try to load output points from CSV
                output_csv_path = os.path.join(base_data_path, "Test2output.csv")
                output_points = []
                if os.path.exists(output_csv_path):
                    try:
                        df_out = pd.read_csv(output_csv_path)
                        # Assuming columns like X, Y, or similar. Adjust based on actual CSV format.
                        # If no header, assume first two cols are X, Y
                        if 'X' in df_out.columns and 'Y' in df_out.columns:
                            for idx, row in df_out.iterrows():
                                output_points.append((row['X'], row['Y'], f"Point_{idx+1}"))
                        else:
                            # Fallback: Read as raw numpy
                            arr_out = df_out.values
                            for i in range(arr_out.shape[0]):
                                output_points.append((arr_out[i, 0], arr_out[i, 1], f"Point_{i+1}"))
                        print(f"Loaded {len(output_points)} output points from CSV.")
                    except Exception as e:
                        print(f"Warning: Could not read Test2output.csv: {e}")
                
                # If no points found (or file missing), define the center points of depressions roughly?
                # The depressions are in a 4x4 matrix in 2000x2000. 
                # Centers approx: 250, 750, 1250, 1750 in both directions.
                if not output_points:
                    print("Using default representative points (Depression centers).")
                    output_points = [
                        (250.0, 250.0, "Depression_1_1"),
                        (1250.0, 1250.0, "Depression_3_3"),
                        (1750.0, 1750.0, "Depression_4_4")
                    ]

                def plot_gauge(x, y, name, filename):
                    pts = jnp.stack([jnp.full_like(t_plot, x), jnp.full_like(t_plot, y), t_plot], axis=-1)
                    U = model.apply(final_params, pts, train=False)
                    h_pred = U[..., 0]
                    
                    plt.figure(figsize=(10, 6))

                    # Plot Baseline if available
                    if full_val_data is not None:
                        # Convert to numpy for flexible boolean indexing
                        val_np = np.array(full_val_data)
                        # Filter for current gauge coordinates
                        mask = np.isclose(val_np[:, 1], x) & np.isclose(val_np[:, 2], y)
                        gauge_data = val_np[mask]
                        
                        if gauge_data.shape[0] > 0:
                            # Sort by time
                            gauge_data = gauge_data[gauge_data[:, 0].argsort()]
                            plt.plot(gauge_data[:, 0], gauge_data[:, 3], 'k--', linewidth=1.5, alpha=0.7, label=f'Baseline {name}')

                    plt.plot(t_plot, h_pred, label=f'Predicted h @ ({x},{y})')
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
    parser = argparse.ArgumentParser(description="Unified PINN training script for SWE (Test 2).")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path: sys.path.insert(0, project_root)

    main(args.config)