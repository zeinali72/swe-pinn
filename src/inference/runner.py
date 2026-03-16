"""Core inference orchestrator.

``run_inference`` is the single entry point that loads a trained model,
generates predictions, computes all applicable metrics, and writes reports.
"""
import os
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from jax import random
from flax.core import FrozenDict

from src.config import load_config
from src.training.setup import (
    init_model_from_config,
    get_experiment_name,
    resolve_experiment_paths,
    resolve_configured_asset_path,
    apply_irregular_domain_bounds,
    apply_output_scales,
    get_data_filename,
)
from src.checkpointing import load_checkpoint
from src.predict import Predictor
from src.metrics.accuracy import compute_validation_metrics
from src.metrics.peak import r_squared, peak_depth_error, time_to_peak_error, rmse_mae_ratio
from src.metrics.negative_depth import negative_depth_stats
from src.metrics.flood_extent import flood_extent_metrics
from src.metrics.conservation import volume_balance, continuity_residual
from src.metrics.boundary import initial_condition_accuracy
from src.metrics.decomposition import spatial_decomposition, temporal_decomposition
from src.inference.context import InferenceContext
from src.inference.experiment_registry import get_experiment_meta
from src.inference import reporting


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_domain_bounds(cfg_dict):
    """Extract domain bounds from config into a plain dict."""
    domain = cfg_dict.get("domain", {})
    return {
        "lx": domain.get("lx", 1.0),
        "ly": domain.get("ly", 1.0),
        "t_final": domain.get("t_final", 1.0),
        "x_min": domain.get("x_min", 0.0),
        "y_min": domain.get("y_min", 0.0),
    }


def _load_validation(cfg_dict, paths_info, meta, experiment_name):
    """Load or generate validation data depending on experiment type."""
    if meta["reference_type"] == "analytical":
        return _generate_analytical_validation(cfg_dict)

    val_file = get_data_filename(cfg_dict, "validation_file",
                                 meta.get("default_val_file", "validation_sample.npy"))
    val_path = os.path.join(paths_info["base_data_path"], val_file)

    if not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Validation data not found at {val_path}. "
            f"Set data.validation_file in config or provide the file."
        )

    from src.data import load_validation_data
    _raw, inputs, targets = load_validation_data(val_path)
    return jnp.array(inputs), jnp.array(targets)


def _generate_analytical_validation(cfg_dict):
    """Generate validation grid + analytical reference for Experiment 1."""
    from src.physics import h_exact, hu_exact, hv_exact

    domain = cfg_dict["domain"]
    physics = cfg_dict["physics"]
    plot_cfg = cfg_dict.get("plotting", {})

    lx = domain["lx"]
    ly = domain["ly"]
    t_final = domain["t_final"]

    nx = plot_cfg.get("nx_val", 101)
    ny = max(int(ly / lx * nx), 5)
    nt = 21

    xs = jnp.linspace(0, lx, nx)
    ys = jnp.linspace(0, ly, ny)
    ts = jnp.linspace(0, t_final, nt)

    xx, yy, tt = jnp.meshgrid(xs, ys, ts, indexing="ij")
    coords = jnp.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=-1)

    n_manning = physics["n_manning"]
    u_const = physics["u_const"]
    h_ref = h_exact(coords[:, 0], coords[:, 2], n_manning, u_const)
    hu_ref = hu_exact(coords[:, 0], coords[:, 2], n_manning, u_const)
    hv_ref = hv_exact(coords[:, 0], coords[:, 2], n_manning, u_const)
    targets = jnp.stack([h_ref, hu_ref, hv_ref], axis=-1)

    return coords, targets


def _load_experiment_assets(cfg_dict, paths_info, meta):
    """Load optional experiment-specific assets (DEM, irregular sampler, BC)."""
    domain_sampler = None
    bc_fn = None
    scenario_name = paths_info["scenario_name"]
    base = paths_info["base_data_path"]

    # Irregular domain sampler (Experiments 7, 8)
    if meta["domain_type"] == "irregular":
        domain_path = resolve_configured_asset_path(
            cfg_dict, base, scenario_name, "domain_artifacts", required=True,
        )
        from src.data import IrregularDomainSampler
        domain_sampler = IrregularDomainSampler(domain_path)
        apply_irregular_domain_bounds(cfg_dict, domain_sampler)
        default_scales = cfg_dict.get("model", {}).get("output_scales", (1.0, 1.0, 1.0))
        apply_output_scales(cfg_dict, default_scales)

    # Bathymetry
    if meta["has_bathymetry"]:
        dem_path = resolve_configured_asset_path(
            cfg_dict, base, scenario_name, "dem", required=False,
        )
        if dem_path and os.path.exists(dem_path):
            from src.data import load_bathymetry
            load_bathymetry(dem_path)

    # Boundary condition function
    bc_path = resolve_configured_asset_path(
        cfg_dict, base, scenario_name, "boundary_condition", required=False,
    )
    if bc_path and os.path.exists(bc_path):
        from src.data import load_boundary_condition
        bc_fn = load_boundary_condition(bc_path)

    return domain_sampler, bc_fn


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_accuracy_metrics(ctx: InferenceContext) -> dict:
    """Core accuracy metrics (always computed)."""
    targets = ctx.val_targets
    preds = ctx.predictions

    # Ensure targets has 3 columns
    if targets.shape[-1] == 1:
        # h-only validation
        acc = compute_validation_metrics(
            preds[..., :1], targets,
        )
        # Replace the per-variable keys to be h-only
        acc_h = {k: v for k, v in acc.items() if k.endswith("_h")}
        acc = acc_h
    else:
        acc = compute_validation_metrics(preds, targets)

    # Peak metrics (on h)
    pred_h = preds[..., 0]
    true_h = targets[..., 0] if targets.shape[-1] >= 1 else targets.ravel()
    t_coords = ctx.val_coords[:, 2]

    acc["r_squared_h"] = r_squared(pred_h, true_h)
    acc["peak_depth_error"] = peak_depth_error(pred_h, true_h)
    acc["time_to_peak_error"] = time_to_peak_error(pred_h, true_h, t_coords)
    acc["rmse_mae_ratio_h"] = rmse_mae_ratio(pred_h, true_h)

    # Negative depth
    acc["negative_depth"] = negative_depth_stats(pred_h)

    return acc


def _compute_optional_metrics(ctx: InferenceContext, *, skip_conservation: bool = False) -> dict:
    """Optional metrics gated by experiment registry and flags."""
    results = {}
    meta = ctx.experiment_meta
    pred_h = ctx.predictions[..., 0]
    targets = ctx.val_targets
    true_h = targets[..., 0] if targets.shape[-1] >= 1 else targets.ravel()

    # Flood extent
    if meta.get("flood_extent"):
        results["flood_extent"] = flood_extent_metrics(pred_h, true_h)

    # Conservation
    if not skip_conservation:
        results["volume_balance"] = volume_balance(
            pred_h, ctx.val_coords, ctx.domain_bounds,
        )
        n_sample = min(10_000, ctx.val_coords.shape[0])
        key = random.PRNGKey(0)
        indices = random.choice(key, ctx.val_coords.shape[0], shape=(n_sample,), replace=False)
        sample_pts = ctx.val_coords[indices]
        results["continuity_residual"] = continuity_residual(
            ctx.model, ctx.params, sample_pts, ctx.config,
        )

    # Spatial decomposition
    if targets.shape[-1] >= 3:
        results["spatial_decomposition"] = spatial_decomposition(
            ctx.predictions, targets, ctx.val_coords, ctx.domain_bounds,
        )

    # Temporal decomposition
    results["temporal_decomposition"] = temporal_decomposition(
        pred_h, true_h, ctx.val_coords[:, 2],
        pred_full=ctx.predictions if targets.shape[-1] >= 3 else None,
        true_full=targets if targets.shape[-1] >= 3 else None,
    )

    # Initial condition accuracy
    t = ctx.val_coords[:, 2]
    t_min = float(jnp.min(t))
    ic_mask = jnp.abs(t - t_min) < 1e-6
    if jnp.sum(ic_mask) > 0:
        ic_coords = ctx.val_coords[ic_mask]
        results["initial_condition"] = initial_condition_accuracy(
            ctx.model, ctx.params, ic_coords,
        )

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference(
    config_path: str,
    checkpoint_path: str,
    output_dir: str = None,
    *,
    checkpoint_name: str = "best_nse",
    skip_plots: bool = False,
    skip_conservation: bool = False,
    batch_size: int = 50_000,
) -> dict:
    """Run the full inference pipeline on a single checkpoint.

    Args:
        config_path: Path to experiment YAML config.
        checkpoint_path: Path to a checkpoint directory (containing model.pkl).
        output_dir: Where to write reports. Auto-generated if None.
        checkpoint_name: Label for this checkpoint (e.g. ``best_nse``).
        skip_plots: If True, skip figure generation.
        skip_conservation: If True, skip expensive autodiff conservation metrics.
        batch_size: Predictor batch size.

    Returns:
        Full results dict.
    """
    # 1. Load config
    cfg_dict = load_config(config_path)
    experiment_name = get_experiment_name(cfg_dict)
    meta = get_experiment_meta(experiment_name)
    paths_info = resolve_experiment_paths(cfg_dict)

    print(f"Inference: experiment={experiment_name}, checkpoint={checkpoint_name}")

    # 2. Load experiment assets
    domain_sampler, bc_fn = _load_experiment_assets(cfg_dict, paths_info, meta)

    # 3. Reconstruct model
    cfg = FrozenDict(cfg_dict)
    model, _params_init, _tk, _vk = init_model_from_config(cfg)

    # 4. Load checkpoint
    params, train_meta = load_checkpoint(checkpoint_path)
    if params is None:
        raise FileNotFoundError(f"No model.pkl found in {checkpoint_path}")

    # 5. Load validation data
    val_coords, val_targets = _load_validation(cfg_dict, paths_info, meta, experiment_name)

    # 6. Predict
    min_depth = cfg_dict.get("numerics", {}).get("min_depth", 0.0)
    predictor = Predictor(model, batch_size=batch_size, min_depth=min_depth)
    predictions, elapsed = predictor.predict_timed(params, val_coords)
    print(f"  Prediction: {val_coords.shape[0]:,} points in {elapsed:.3f}s")

    # 7. Build context
    domain_bounds = _build_domain_bounds(cfg_dict)
    ctx = InferenceContext(
        config=cfg_dict,
        model=model,
        params=params,
        predictor=predictor,
        val_coords=val_coords,
        val_targets=val_targets,
        predictions=predictions,
        experiment_name=experiment_name,
        domain_bounds=domain_bounds,
        experiment_meta=meta,
        inference_time_seconds=elapsed,
        training_metadata=train_meta,
        domain_sampler=domain_sampler,
        bc_fn=bc_fn,
        checkpoint_name=checkpoint_name,
    )

    # 8. Compute metrics
    results = {"checkpoint": checkpoint_name}
    results["accuracy"] = _compute_accuracy_metrics(ctx)
    results["inference_time_seconds"] = elapsed
    results["n_points"] = int(val_coords.shape[0])

    optional = _compute_optional_metrics(ctx, skip_conservation=skip_conservation)
    results.update(optional)

    if train_meta:
        results["training_metadata"] = train_meta

    # 9. Generate reports
    if output_dir is None:
        output_dir = os.path.join("inference_output", experiment_name, checkpoint_name)
    os.makedirs(output_dir, exist_ok=True)

    reporting.save_yaml_summary(results, output_dir)
    reporting.save_raw_predictions(ctx, output_dir)
    reporting.print_text_report(results, output_dir)

    if "spatial_decomposition" in results:
        reporting.save_csv_tables(results, output_dir)

    if not skip_plots:
        reporting.generate_plots(ctx, results, output_dir)

    print(f"  Reports written to {output_dir}")
    return results


def run_inference_multi(
    config_path: str,
    checkpoint_dir: str,
    output_dir: str = None,
    *,
    checkpoints: str = "all",
    **kwargs,
) -> dict:
    """Evaluate multiple checkpoints and produce a comparison table.

    Args:
        config_path: Path to experiment YAML config.
        checkpoint_dir: Parent directory containing ``best_nse/``, ``best_loss/``, ``final/``.
        output_dir: Where to write the combined report.
        checkpoints: ``"all"`` or comma-separated list of checkpoint names.
        **kwargs: Forwarded to ``run_inference``.

    Returns:
        Dict mapping checkpoint name to its results dict.
    """
    if checkpoints == "all":
        names = ["best_nse", "best_loss", "final"]
    else:
        names = [n.strip() for n in checkpoints.split(",")]

    ckpt_dir = Path(checkpoint_dir)
    all_results = {}
    for name in names:
        ckpt_path = ckpt_dir / name
        if not ckpt_path.exists():
            print(f"  Skipping {name}: {ckpt_path} does not exist")
            continue
        sub_output = os.path.join(output_dir or "inference_output", name) if output_dir else None
        all_results[name] = run_inference(
            config_path, str(ckpt_path),
            output_dir=sub_output,
            checkpoint_name=name,
            **kwargs,
        )

    if output_dir and len(all_results) > 1:
        reporting.save_comparison_table(all_results, output_dir)

    return all_results
