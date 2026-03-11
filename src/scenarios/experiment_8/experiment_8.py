"""Experiment 8 — Real urban domain, Eastbourne UK (Phase 3).

Applies the framework to a real urban subcatchment (Blue Heart Project).
Buildings excluded from mesh by construction, treated as wall boundaries.
Requires: configs/experiment_8.yaml, data/experiment_8/
Builds on: Experiment 7.
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
import pandas as pd 

from jaxtyping import config

import jax
import jax.numpy as jnp
from jax import random, lax
import optax
try:
    from aim import Image
except ImportError:
    Image = None
from flax.core import FrozenDict
import numpy as np 
import matplotlib
matplotlib.use('Agg') # Ensure headless plotting
import matplotlib.pyplot as plt 

# Local application imports
from src.config import load_config, DTYPE
from src.data import (
    get_batches_tensor,
    get_sample_count,
    load_boundary_condition,
    IrregularDomainSampler,
    load_bathymetry,
    load_validation_data,
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

from src.monitoring import ConsoleLogger, AimTracker, compute_negative_depth_diagnostics
from src.metrics.accuracy import compute_validation_metrics
from src.checkpointing import CheckpointManager

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
        
        # A. Upstream Boundary (Flux prescribed)
        # CHANGED: Using 'bc_upstream' to match preprocess output
        if batch['bc_upstream'].shape[0] > 0:
            t_inflow = batch['bc_upstream'][..., 2]
            Q_target = bc_fn_static(t_inflow) # m^3/s
            
            # Assuming 100m width for Experiment 8 upstream boundary
            flux_target_x = Q_target / 372.92  # m^2/s 
            
            loss_inflow_x = loss_boundary_dirichlet_hu(model, params, batch['bc_upstream'], flux_target_x)
            loss_inflow_y = loss_boundary_dirichlet_hv(model, params, batch['bc_upstream'], jnp.zeros_like(flux_target_x))
            loss_bc_inflow = loss_inflow_x + loss_inflow_y
        else:
            loss_bc_inflow = 0.0

        # B. Wall Boundaries (Generalized Slip) - Outer Domain
        loss_bc_wall = loss_slip_wall_generalized(model, params, batch['bc_wall'])
        
        # C. Building Boundaries (Generalized Slip) - Obstacles
        loss_bldg = loss_slip_wall_generalized(model, params, batch['bc_building'])
        
        total_bc = loss_bc_inflow + loss_bc_wall + loss_bldg

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
            'data': loss_data,
            # 'building' is now part of 'bc' sum, but can be tracked separately if needed in reporting
            'building': loss_bldg 
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
    Main training loop for Experiment 8 Scenario.
    """
    #--- 1. LOAD CONFIGURATION (MUTABLE) ---
    cfg_dict = load_config(config_path)
    
    print("Info: Running Experiment 8 training...")

    # --- 2. SETUP DATA & COMPUTE DOMAIN EXTENT ---
    scenario_name = cfg_dict.get('scenario', 'experiment_8')
    base_data_path = os.path.join("data", scenario_name)

    # A. Init Irregular Domain Sampler
    artifacts_path = os.path.join(base_data_path, "domain_artifacts.npz")
    if not os.path.exists(artifacts_path):
        artifacts_path = os.path.join(base_data_path, "domain.npz")
        
    if not os.path.exists(artifacts_path):
        print(f"Error: Domain artifacts file not found at {artifacts_path}")
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

    # Standard scaling for this experiment
    h_scale = 1.0  
    hu_scale = 1.0 
    hv_scale = 1.0

    if 'model' not in cfg_dict: cfg_dict['model'] = {}
    cfg_dict['model']['output_scales'] = (h_scale, hu_scale, hv_scale)
    
    print(f"Active Output Scaling: {cfg_dict['model']['output_scales']}")

    # --- 3. FINALIZE CONFIG & INIT MODEL ---
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
    dem_path = os.path.join(base_data_path, "DEM_v2_asc.asc")
    if not os.path.exists(dem_path):
        print(f"Error: DEM file not found at {dem_path}")
        sys.exit(1)
    print(f"Loading Bathymetry from {dem_path}...")
    load_bathymetry(dem_path)
    
    # C. Load Boundary Condition Function
    bc_csv_path = os.path.join(base_data_path, "Test6_BC_interpolated.csv")
    if not os.path.exists(bc_csv_path):
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
            print("Warning: 'data_free' flag not specified. Defaulting to 'data_free: true'.")
        else:
            print("Info: 'data_free: true' found in config. Data loss term disabled.")
        has_data_loss = False
        data_free = True

    # Training Data (for Data Loss)
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

    # E. Load Validation Data (Ground Truth)
    validation_data_file = os.path.join(base_data_path, "validation_gauges_ground_truth.npy")
    validation_data_loaded = False
    
    # Validation arrays
    val_pts_batch = None
    val_h_true = None
    val_hu_true = None
    val_hv_true = None
    
    if os.path.exists(validation_data_file):
        try:
            print(f"Loading VALIDATION data from: {validation_data_file}")
            _, val_pts_batch, val_targets = load_validation_data(validation_data_file, dtype=DTYPE)
            val_h_true = val_targets[:, 0]
            u_temp = val_targets[:, 1]
            v_temp = val_targets[:, 2]
            
            val_hu_true = val_h_true * u_temp
            val_hv_true = val_h_true * v_temp
            
            if val_pts_batch.shape[0] > 0:
                validation_data_loaded = True
                print(f"Validation loaded: {val_pts_batch.shape[0]} points. Targets: h, hu, hv prepared.")
            else:
                 print("Warning: Validation data file is empty.")
        except Exception as e:
            print(f"Error loading validation data: {e}")
            val_pts_batch = None
    else:
        print(f"Warning: Validation data not found at {validation_data_file}.")

    # --- 7. Initialize Aim & Console Logger ---
    aim_enabled = cfg_dict.get('aim', {}).get('enable', True)
    aim_tracker = AimTracker(cfg_dict, trial_name, enable=aim_enabled)
    aim_tracker.log_flags({"scenario_type": "experiment_8"})
    if aim_enabled:
        try:
            aim_tracker.log_artifact(config_path, 'run_config.yaml')
            aim_tracker.log_artifact(os.path.abspath(__file__), 'source_script.py')
        except Exception:
            pass

    # --- 8. Summary ---
    cfg_dict['scenario'] = cfg_dict.get('scenario', 'experiment_8')
    console = ConsoleLogger(cfg_dict)
    console.print_header()

    start_time = time.time()
    ckpt_mgr = CheckpointManager(model_dir, model=model)
    val_metrics = {}
    neg_depth = {}
    avg_losses_unweighted = {}
    avg_total_weighted_loss = 0.0
    global_step = 0

    # --- 9. Data Generation Setup ---
    sampling_cfg = cfg["sampling"]
    batch_size = cfg["training"]["batch_size"]
    domain_cfg = cfg["domain"]
    
    n_pde = get_sample_count(sampling_cfg, "n_points_pde", 1000)
    n_ic = get_sample_count(sampling_cfg, "n_points_ic", 100)
    # Using 'n_points_bc_inflow' for upstream sampling
    n_bc_upstream = get_sample_count(sampling_cfg, "n_points_bc_inflow", 100) 
    n_bc_wall = get_sample_count(sampling_cfg, "n_points_bc_domain", 100)
    n_building = get_sample_count(sampling_cfg, "n_points_bc_building", 100)

    bc_counts = [n_pde//batch_size, n_ic//batch_size, n_bc_wall//batch_size, n_bc_upstream//batch_size, n_building//batch_size]
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
        # CHANGED: Sampling 'upstream' boundary instead of generic 'inflow'
        bc_upstream_pts = domain_sampler.sample_boundary(k3, n_bc_upstream, (0., domain_cfg["t_final"]), 'upstream')
        bc_upstream = get_batches_tensor(k3, bc_upstream_pts, batch_size, num_batches)

        bc_wall_pts = domain_sampler.sample_boundary(k4, n_bc_wall, (0., domain_cfg["t_final"]), 'wall')
        bc_wall = get_batches_tensor(k4, bc_wall_pts, batch_size, num_batches)
        
        bc_building_pts = domain_sampler.sample_boundary(k5, n_building, (0., domain_cfg["t_final"]), 'building')
        bc_building = get_batches_tensor(k5, bc_building_pts, batch_size, num_batches)
        
        # Data
        if not data_free and data_points_full is not None:
             data_d = get_batches_tensor(k5, data_points_full, batch_size, num_batches)
        else:
             data_d = jnp.zeros((num_batches, 0, 6), dtype=DTYPE)

        return {
            'pde': pde_data, 'ic': ic_data, 
            'bc_upstream': bc_upstream,
            'bc_wall': bc_wall, 
            'bc_building': bc_building,
            'data': data_d
        }
    
    generate_epoch_data_jitted = jax.jit(generate_epoch_data)

    # Scan Body
    def scan_body(carry, batch_data):
        curr_params, curr_opt_state = carry
        current_all_batches = {
            'pde': batch_data['pde'],
            'ic': batch_data['ic'],
            'bc_upstream': batch_data['bc_upstream'],
            'bc_wall': batch_data['bc_wall'], 
            'bc_building': batch_data['bc_building'],
            'data': batch_data['data']
        }
        new_params, new_opt_state, terms, total = train_step_jitted(
            model, optimiser, curr_params, curr_opt_state,
            current_all_batches, cfg, data_free, bc_fn_static, current_weights_dict
        )
        return (new_params, new_opt_state), (terms, total)

    # --- 10. Training Loop ---
    # Tracking Metric: Combined NSE (NSE_h + NSE_hu + NSE_hv)
    best_nse_stats = {
        'combined_nse': -jnp.inf, 
        'nse_h': -jnp.inf,
        'nse_hu': -jnp.inf,
        'nse_hv': -jnp.inf,
        'epoch': 0, 'global_step': 0,
        'time_elapsed_seconds': 0.0, 'total_weighted_loss': 0.0, 'unweighted_losses': {}
    }
    
    best_loss_stats = {
        'total_weighted_loss': jnp.inf, 'epoch': 0, 'global_step': 0,
        'time_elapsed_seconds': 0.0, 'combined_nse': -jnp.inf, 'rmse': jnp.inf, 'unweighted_losses': {}
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

            
            # --- Validation (h + hu + hv) ---
            combined_nse_val = -float('inf')
            nse_h_val, nse_hu_val, nse_hv_val = -float('inf'), -float('inf'), -float('inf')
            rmse_val = float('inf')

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
            nse_val, rmse_val = -float('inf'), float('inf')
            if validation_data_loaded:
                try:
                    U_pred = model.apply(params, val_pts_batch, train=False)

                    h_pred = U_pred[..., 0]
                    hu_pred = U_pred[..., 1]
                    hv_pred = U_pred[..., 2]

                    nse_h_val = float(nse(val_h_true, h_pred))
                    nse_hu_val = float(nse(val_hu_true, hu_pred))
                    nse_hv_val = float(nse(val_hv_true, hv_pred))

                    combined_nse_val = (nse_h_val + nse_hu_val + nse_hv_val)/3.0

                    rmse_val = float(rmse(val_h_true, h_pred))

                except Exception as e:
                    print(f"Validation Error: {e}")

            val_metrics = {
                'nse_h': float(nse_h_val), 'nse_hu': float(nse_hu_val), 'nse_hv': float(nse_hv_val),
                'rmse_h': float(rmse_val), 'combined_nse': float(combined_nse_val)
            }

            # --- Update Best Model Statistics ---
            if combined_nse_val > best_nse_stats['combined_nse']:
                best_nse_stats.update({
                    'combined_nse': combined_nse_val,
                    'nse_h': nse_h_val,
                    'nse_hu': nse_hu_val,
                    'nse_hv': nse_hv_val,
                    'rmse': rmse_val, 'epoch': epoch, 'global_step': global_step,
                    'time_elapsed_seconds': time.time() - start_time,
                    'total_weighted_loss': avg_total_weighted_loss,
                    'unweighted_losses': {k: float(v) for k, v in avg_losses_unweighted.items()}
                })
                best_params_nse = copy.deepcopy(params)
                if combined_nse_val > -jnp.inf:
                    print(f"    ---> New Best Combined NSE: {combined_nse_val:.4f} (h:{nse_h_val:.4f} hu:{nse_hu_val:.4f} hv:{nse_hv_val:.4f})")

            if avg_total_weighted_loss < best_loss_stats['total_weighted_loss']:
                best_loss_stats.update({
                    'total_weighted_loss': avg_total_weighted_loss, 'epoch': epoch, 'global_step': global_step,
                    'time_elapsed_seconds': time.time() - start_time,
                    'unweighted_losses': {k: float(v) for k, v in avg_losses_unweighted.items()},
                    'combined_nse': combined_nse_val,
                    'nse_h': nse_h_val,
                    'nse_hu': nse_hu_val,
                    'nse_hv': nse_hv_val,
                    'rmse': rmse_val
                })
                best_params_loss = copy.deepcopy(params)

            # Negative depth diagnostics and checkpoint tracking
            freq = cfg.get("reporting", {}).get("epoch_freq", 100)
            epoch_time = time.time() - epoch_start_time

            neg_depth = {'count': 0, 'fraction': 0.0, 'min': 0.0, 'mean': 0.0}
            if (epoch + 1) % freq == 0:
                try:
                    neg_depth = compute_negative_depth_diagnostics(model, params, scan_inputs['pde'][0])
                except Exception:
                    pass

            saved_events = ckpt_mgr.update(
                epoch, params, opt_state, val_metrics,
                avg_losses_unweighted, avg_total_weighted_loss, cfg_dict, neg_depth
            )
            for event in saved_events:
                event_type, value, ep, prev_value, prev_epoch = event
                if event_type == 'best_nse':
                    console.print_checkpoint_nse(value, ep, prev_value, prev_epoch)
                    aim_tracker.log_best_nse(value, ep)
                elif event_type == 'best_loss':
                    console.print_checkpoint_loss(value, ep, prev_value, prev_epoch)
                    aim_tracker.log_best_loss(value, ep)

            # Reporting
            if (epoch + 1) % freq == 0:
                console.print_epoch(
                    epoch, cfg["training"]["epochs"],
                    avg_losses_unweighted, avg_total_weighted_loss,
                    current_lr, 0.0,
                    val_metrics, neg_depth.get('fraction', 0.0), epoch_time
                )
                if validation_data_loaded:
                    print(f"    Val NSE Breakdown: Combined={combined_nse_val:.4f} | h={nse_h_val:.4f} hu={nse_hu_val:.4f} hv={nse_hv_val:.4f}")

            aim_tracker.log_epoch(
                epoch=epoch, step=global_step,
                losses=avg_losses_unweighted, total_loss=avg_total_weighted_loss,
                val_metrics=val_metrics, lr=current_lr, grad_norm=0.0,
                epoch_time=epoch_time, elapsed_time=time.time() - start_time,
                neg_depth=neg_depth if (epoch + 1) % freq == 0 else None,
            )

            # --- Early Stopping Check ---
            min_epochs = cfg.get("device", {}).get("early_stop_min_epochs", float('inf'))
            patience = cfg.get("device", {}).get("early_stop_patience", float('inf'))

            if epoch >= min_epochs and (epoch - best_nse_stats['epoch']) >= patience:
                print(f"--- Early stopping triggered at epoch {epoch+1} ---")
                print(f"Best NSE {best_nse_stats['combined_nse']:.6f} achieved at epoch {best_nse_stats['epoch']+1}.")
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

        ckpt_mgr.save_final(epoch if 'epoch' in locals() else 0, params, opt_state, val_metrics, avg_losses_unweighted, avg_total_weighted_loss, cfg_dict, neg_depth)

        best_nse_ckpt = ckpt_mgr.get_best_nse_stats()
        best_loss_ckpt = ckpt_mgr.get_best_loss_stats()

        console.print_completion_summary(
            total_time=total_time,
            final_epoch=epoch if 'epoch' in locals() else 0,
            best_nse_stats=best_nse_ckpt,
            best_loss_stats=best_loss_ckpt,
            final_losses=avg_losses_unweighted,
            final_val_metrics=val_metrics,
            neg_depth_final=neg_depth,
            neg_depth_best_nse={},
            neg_depth_best_loss={},
            final_lr=current_lr if 'current_lr' in locals() else cfg["training"]["learning_rate"],
        )

        final_params = best_params_loss if best_params_loss is not None else best_params_nse

        aim_tracker.log_summary({
            'best_validation_model': best_nse_stats,
            'best_loss_model': best_loss_stats,
            'final_system': {
                'total_training_time_seconds': total_time,
                'total_epochs_run': (epoch + 1) if 'epoch' in locals() else 0,
                'total_steps_run': global_step
            }
        })

        if ask_for_confirmation():
            if final_params is not None:
                saved_model_path = save_model(final_params, model_dir, trial_name)

                if aim_tracker.enabled and saved_model_path:
                    aim_tracker.log_artifact(saved_model_path, 'model_weights.pkl')

                print("Generating Experiment 8 plots...")
                t_plot = jnp.arange(0., cfg['domain']['t_final'], 60.0, dtype=DTYPE)

                output_csv_path = os.path.join(base_data_path, "Test6output.csv")
                output_points = []
                if os.path.exists(output_csv_path):
                    try:
                        df_out = pd.read_csv(output_csv_path)
                        df_out.columns = [c.strip() for c in df_out.columns]
                        if 'X' in df_out.columns and 'Y' in df_out.columns:
                            for idx, row in df_out.iterrows():
                                output_points.append((row['X'], row['Y'], f"Point_{idx+1}"))
                        else:
                            arr_out = df_out.values
                            for i in range(arr_out.shape[0]):
                                output_points.append((arr_out[i, 0], arr_out[i, 1], f"Point_{i+1}"))
                        print(f"Loaded {len(output_points)} output points from CSV.")
                    except Exception as e:
                        print(f"Warning: Could not read Test6output.csv: {e}")

                if not output_points:
                    cx, cy = (x_max+x_min)/2, (y_max+y_min)/2
                    output_points = [(cx, cy, "Center_Point")]

                def plot_gauge(x, y, name, filename):
                    pts = jnp.stack([jnp.full_like(t_plot, x), jnp.full_like(t_plot, y), t_plot], axis=-1)
                    U = model.apply(final_params, pts, train=False)
                    h_pred = U[..., 0]

                    plt.figure(figsize=(10, 6))

                    gauge_data = None
                    if validation_data_loaded:
                        dists = jnp.sqrt((loaded_val_data[:, 1] - x)**2 + (loaded_val_data[:, 2] - y)**2)
                        mask = dists < 2.0
                        subset = loaded_val_data[mask]
                        if subset.shape[0] > 0:
                            gauge_data = subset

                    if gauge_data is not None and gauge_data.shape[0] > 0:
                        gauge_data = gauge_data[jnp.argsort(gauge_data[:, 0])]
                        gd_t = np.array(gauge_data[:, 0])
                        gd_h = np.array(gauge_data[:, 3])
                        plt.plot(gd_t, gd_h, 'k--', linewidth=1.5, alpha=0.7, label=f'Baseline {name}')

                    tp_np = np.array(t_plot)
                    hp_np = np.array(h_pred)

                    plt.plot(tp_np, hp_np, label=f'Predicted h @ ({x:.1f},{y:.1f})')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Water Level h (m)')
                    plt.title(f'{name} - Water Level vs Time')
                    plt.legend()
                    plt.grid(True)
                    path = os.path.join(results_dir, filename)
                    plt.savefig(path)
                    plt.close()
                    aim_tracker.log_image(path, filename, epoch if 'epoch' in locals() else 0)

                for px, py, pname in output_points:
                     plot_gauge(px, py, pname, f"{pname}_timeseries.png")

                print(f"Plots saved to {results_dir}")
            else:
                print("No model parameters found to save.")

        else:
            print("Save aborted by user. Deleting artifacts...")
            try:
                aim_tracker.delete_run()

                if os.path.exists(results_dir):
                    shutil.rmtree(results_dir)
                    print(f"Deleted results directory: {results_dir}")
                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                    print(f"Deleted model directory: {model_dir}")

                print("Cleanup complete.")
            except Exception as e:
                print(f"Error during cleanup: {e}")

        aim_tracker.close()

    return best_nse_stats['combined_nse']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified PINN training script for SWE (Experiment 8 - Irregular).")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path: sys.path.insert(0, project_root)

    main(args.config)