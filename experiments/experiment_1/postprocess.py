"""Experiment 1 postprocessing pipeline — analytical dam-break verification.

Loads a trained checkpoint and config, generates the analytical validation
grid, computes all applicable metrics, and produces all applicable plots.

Applicable metrics  (per experimental programme reference):
  A1-A4 : NSE, RMSE, MAE, Rel L2  (h, hu, hv)
  B1    : Domain-integral volume balance
  C1    : Slip violation at top/bottom walls
  C2    : Inflow boundary error at x=0
  C3    : Outflow zero-gradient residual at x=lx  (Exp 1 only)
  C4    : Initial condition error at t=0
  D2    : Training cost (read from checkpoint metadata if available)
  D3    : Inference cost (timed forward pass)

Applicable plots:
  P1.1  : Gauge time series (h vs t at representative x locations)
  P1.2  : Mass balance time series
  P1.3  : Training loss curves (if history file present)
  P1.4  : Validation NSE during training (if history file present)
  P2.1  : Spatial error map at 3-5 time steps
  P2.2  : PINN vs analytical depth map at mid-time
  P2.7  : Spatial error decomposition (shock / boundary / interior)
  P3.1  : Precision comparison bar (if multi-precision results provided)

Usage
-----
python -m experiments.experiment_1.postprocess \\
    --config configs/experiment_1.yaml \\
    --checkpoint models/experiment_1/best_nse \\
    [--output inference_output/experiment_1/best_nse] \\
    [--precision-results float64=path1,float32=path2,bfloat16=path3] \\
    [--training-history models/experiment_1/training_history.json] \\
    [--batch-size 50000] \\
    [--skip-conservation] \\
    [--skip-plots]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from src.config import load_config
from src.training.setup import init_model_from_config, get_experiment_name
from src.checkpointing import load_checkpoint
from src.physics import h_exact, hu_exact, hv_exact

from evaluation.metrics.accuracy import compute_all_accuracy
from evaluation.metrics.boundary import (
    slip_violation,
    inflow_boundary_error,
    outflow_gradient_residual,
    initial_condition_error,
)
from evaluation.metrics.cost import inference_cost
from evaluation.decomposition.spatial_decomp import (
    classify_points,
    compute_spatial_decomposition,
)

# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def _build_validation_grid(cfg_dict: dict) -> tuple:
    """Generate a dense structured (x, y, t) validation grid.

    Returns
    -------
    coords : jnp.ndarray  shape (N, 3)  [x, y, t]
    targets : jnp.ndarray shape (N, 3)  [h, hu, hv]
    xs, ys, ts : 1-D linspaces used to build the grid
    """
    domain = cfg_dict["domain"]
    physics = cfg_dict["physics"]
    plot_cfg = cfg_dict.get("plotting", {})

    lx = domain["lx"]
    ly = domain["ly"]
    t_final = domain["t_final"]

    nx = plot_cfg.get("nx_val", 101)
    ny = max(int(round(ly / lx * nx)), 5)
    nt = 21  # 21 time snapshots over [0, T]

    xs = jnp.linspace(0.0, lx, nx)
    ys = jnp.linspace(0.0, ly, ny)
    ts = jnp.linspace(0.0, t_final, nt)

    xx, yy, tt = jnp.meshgrid(xs, ys, ts, indexing="ij")
    coords = jnp.stack([xx.ravel(), yy.ravel(), tt.ravel()], axis=-1)

    n_manning = physics["n_manning"]
    u_const = physics["u_const"]
    h_ref = h_exact(coords[:, 0], coords[:, 2], n_manning, u_const)
    hu_ref = hu_exact(coords[:, 0], coords[:, 2], n_manning, u_const)
    hv_ref = hv_exact(coords[:, 0], coords[:, 2], n_manning, u_const)
    targets = jnp.stack([h_ref, hu_ref, hv_ref], axis=-1)

    return coords, targets, xs, ys, ts


def _build_boundary_points(cfg_dict: dict, n_pts: int = 500) -> dict:
    """Generate boundary evaluation points for metric computation.

    Returns a dict with keys: 'left', 'right', 'bottom', 'top', 'ic'.
    Each value is an (M, 3) array of [x, y, t] coordinates.
    """
    domain = cfg_dict["domain"]
    lx = domain["lx"]
    ly = domain["ly"]
    t_final = domain["t_final"]

    rng = np.random.default_rng(0)
    t_samples = rng.uniform(0.0, t_final, n_pts)
    y_samples = rng.uniform(0.0, ly, n_pts)
    x_samples = rng.uniform(0.0, lx, n_pts)

    left = np.stack([np.zeros(n_pts), y_samples, t_samples], axis=-1)
    right = np.stack([np.full(n_pts, lx), y_samples, t_samples], axis=-1)
    bottom = np.stack([x_samples, np.zeros(n_pts), t_samples], axis=-1)
    top = np.stack([x_samples, np.full(n_pts, ly), t_samples], axis=-1)
    ic = np.stack([x_samples, y_samples, np.zeros(n_pts)], axis=-1)

    return {
        "left": jnp.asarray(left, dtype=jnp.float32),
        "right": jnp.asarray(right, dtype=jnp.float32),
        "bottom": jnp.asarray(bottom, dtype=jnp.float32),
        "top": jnp.asarray(top, dtype=jnp.float32),
        "ic": jnp.asarray(ic, dtype=jnp.float32),
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _predict_batched(model, params, coords: jnp.ndarray, batch_size: int = 50_000) -> jnp.ndarray:
    """Run model.apply in chunks and concatenate."""
    flax_params = {"params": params["params"]}
    n = coords.shape[0]
    chunks = []
    for start in range(0, n, batch_size):
        out = model.apply(flax_params, coords[start: start + batch_size], train=False)
        chunks.append(out)
    return jnp.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_accuracy(predictions: jnp.ndarray, targets: jnp.ndarray) -> dict:
    pred_dict = {"h": np.asarray(predictions[:, 0]),
                 "hu": np.asarray(predictions[:, 1]),
                 "hv": np.asarray(predictions[:, 2])}
    ref_dict = {"h": np.asarray(targets[:, 0]),
                "hu": np.asarray(targets[:, 1]),
                "hv": np.asarray(targets[:, 2])}
    return compute_all_accuracy(pred_dict, ref_dict)




def _compute_boundary_metrics(model, params, bpts: dict, cfg_dict: dict,
                               skip_conservation: bool = False) -> dict:
    domain = cfg_dict["domain"]
    physics = cfg_dict["physics"]
    eps = cfg_dict.get("numerics", {}).get("eps", 1e-6)
    n_manning = physics["n_manning"]
    u_const = physics["u_const"]
    results = {}

    # C1: slip at top and bottom walls (horizontal walls, normal = ±y)
    wall_pts = jnp.concatenate([bpts["bottom"], bpts["top"]], axis=0)
    n_bot = np.tile([0.0, -1.0], (bpts["bottom"].shape[0], 1))
    n_top = np.tile([0.0,  1.0], (bpts["top"].shape[0], 1))
    wall_normals = jnp.asarray(np.concatenate([n_bot, n_top], axis=0))
    results["C1_slip_violation"] = slip_violation(
        model, params, wall_pts, wall_normals, eps,
    )

    # C2: inflow boundary error at x=0
    t_left = bpts["left"][:, 2]
    h_presc = h_exact(jnp.zeros_like(t_left), t_left, n_manning, u_const)
    results["C2_inflow_error"] = inflow_boundary_error(
        model, params, bpts["left"], np.asarray(h_presc), np.asarray(t_left),
    )

    # C3: outflow zero-gradient at x=lx (Exp 1 only)
    if not skip_conservation:
        results["C3_outflow_gradient"] = outflow_gradient_residual(
            model, params, bpts["right"], cfg_dict,
        )

    # C4: initial condition error at t=0
    results["C4_ic_error"] = initial_condition_error(model, params, bpts["ic"])

    return results


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _generate_plots(
    coords: jnp.ndarray,
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    ts: jnp.ndarray,
    cfg_dict: dict,
    plots_dir: str,
    training_history: dict | None,
    skip_plots: bool,
) -> None:
    if skip_plots:
        return

    os.makedirs(plots_dir, exist_ok=True)

    from evaluation.plots.time_series import (
        plot_gauge_timeseries,
        plot_mass_balance_timeseries,
        plot_training_loss_curves,
        plot_validation_nse_during_training,
    )
    from evaluation.plots.spatial_maps import (
        plot_error_map,
        plot_error_maps_multi,
        plot_depth_map,
        plot_error_decomposition,
    )

    domain = cfg_dict["domain"]
    lx = domain["lx"]
    ly = domain["ly"]
    t_final = domain["t_final"]
    domain_bounds = {"lx": lx, "ly": ly, "t_final": t_final}

    coords_np = np.asarray(coords)
    pred_np = np.asarray(predictions)
    ref_np = np.asarray(targets)
    t_all = coords_np[:, 2]

    # ------------------------------------------------------------------
    # P1.1 — Gauge time series at 3 representative x locations
    # ------------------------------------------------------------------
    gauge_xs = [lx * 0.25, lx * 0.50, lx * 0.75]
    y_mid = ly / 2.0
    n_manning = cfg_dict["physics"]["n_manning"]
    u_const = cfg_dict["physics"]["u_const"]

    from evaluation.metrics.accuracy import compute_nse
    for gx in gauge_xs:
        # Build a dense time vector for this gauge
        t_gauge = np.linspace(0.0, t_final, 200)
        h_ref_gauge = np.asarray(h_exact(
            jnp.full(200, gx), jnp.asarray(t_gauge), n_manning, u_const
        ))

        # Predict at this gauge
        flax_params = {"params": predictions}  # will fail — use model later
        # Use nearest-x points from the validation grid
        mask_gauge = (np.abs(coords_np[:, 0] - gx) < lx * 0.01) & \
                     (np.abs(coords_np[:, 1] - y_mid) < ly * 0.15)
        if mask_gauge.sum() < 3:
            continue
        t_pts = coords_np[mask_gauge, 2]
        h_pred_pts = pred_np[mask_gauge, 0]
        sort_idx = np.argsort(t_pts)
        t_pts = t_pts[sort_idx]
        h_pred_pts = h_pred_pts[sort_idx]
        h_ref_pts = np.asarray(h_exact(
            jnp.full_like(jnp.asarray(t_pts), gx),
            jnp.asarray(t_pts), n_manning, u_const
        ))

        gauge_nse = float(compute_nse(h_pred_pts, h_ref_pts))
        fig = plot_gauge_timeseries(
            t=t_pts,
            predictions={"PINN": h_pred_pts},
            h_ref=h_ref_pts,
            gauge_name=f"x={gx:.0f}m, y={y_mid:.0f}m",
            metrics={"NSE": gauge_nse},
            save_path=os.path.join(plots_dir, f"P1_1_gauge_x{int(gx)}.png"),
        )

    # ------------------------------------------------------------------
    # P1.2 — Mass balance time series
    # ------------------------------------------------------------------
    try:
        t_bins = np.linspace(0.0, t_final, 21)
        t_mids, volumes = [], []
        for i in range(len(t_bins) - 1):
            mask_t = (t_all >= t_bins[i]) & (t_all < t_bins[i + 1])
            if mask_t.sum() > 0:
                mean_h = float(pred_np[mask_t, 0].mean())
                vol = mean_h * lx * ly
            else:
                vol = 0.0
            t_mids.append(float((t_bins[i] + t_bins[i + 1]) / 2))
            volumes.append(vol)

        initial_vol = volumes[0] if volumes[0] > 0 else 1.0
        e_mass = [abs(v - volumes[0]) / max(initial_vol, 1e-12) * 100.0
                  for v in volumes]

        fig = plot_mass_balance_timeseries(
            t_pinn=np.array(t_mids),
            e_mass_pinn=np.array(e_mass),
            save_path=os.path.join(plots_dir, "P1_2_mass_balance.png"),
        )
    except Exception as exc:
        print(f"  Warning: P1.2 failed: {exc}")

    # ------------------------------------------------------------------
    # P1.3 — Training loss curves (from history)
    # ------------------------------------------------------------------
    if training_history is not None:
        try:
            epochs = np.array(training_history.get("epochs", []))
            losses_dict = {k: np.array(v) for k, v in training_history.items()
                           if k.endswith("_loss") or k in
                           ("total", "pde", "bc", "ic", "data", "neg_h")}
            lrs = np.array(training_history["learning_rate"]) \
                if "learning_rate" in training_history else None

            if len(epochs) > 0 and losses_dict:
                fig = plot_training_loss_curves(
                    epochs=epochs,
                    losses_dict=losses_dict,
                    learning_rates=lrs,
                    save_path=os.path.join(plots_dir, "P1_3_loss_curves.png"),
                )
        except Exception as exc:
            print(f"  Warning: P1.3 failed: {exc}")

    # ------------------------------------------------------------------
    # P1.4 — Validation NSE during training
    # ------------------------------------------------------------------
    if training_history is not None:
        try:
            val_epochs = np.array(training_history.get("val_epochs", []))
            nse_h = training_history.get("nse_h", [])
            if len(val_epochs) > 0 and len(nse_h) > 0:
                nse_dict = {"h": np.array(nse_h)}
                if "nse_hu" in training_history:
                    nse_dict["hu"] = np.array(training_history["nse_hu"])
                if "nse_hv" in training_history:
                    nse_dict["hv"] = np.array(training_history["nse_hv"])
                fig = plot_validation_nse_during_training(
                    epochs=val_epochs,
                    nse_dict=nse_dict,
                    save_path=os.path.join(plots_dir, "P1_4_nse_training.png"),
                )
        except Exception as exc:
            print(f"  Warning: P1.4 failed: {exc}")

    # ------------------------------------------------------------------
    # P2 — Spatial plots at 5 time snapshots
    # ------------------------------------------------------------------
    t_snap_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    t_snaps = [t_final * f for f in t_snap_fracs]
    snapshots_error = []
    snapshots_depth_pred = []
    snapshots_depth_ref = []

    for t_s in t_snaps:
        mask_t = np.abs(t_all - t_s) < (t_final / 40.0)
        # Fall back to closest time slice
        if mask_t.sum() < 10:
            idx_closest = np.argmin(np.abs(np.unique(t_all) - t_s))
            t_close = np.unique(t_all)[idx_closest]
            mask_t = np.abs(t_all - t_close) < 1e-3
        if mask_t.sum() < 5:
            continue

        x_s = coords_np[mask_t, 0]
        y_s = coords_np[mask_t, 1]
        err = np.abs(pred_np[mask_t, 0] - ref_np[mask_t, 0])
        h_pred_s = pred_np[mask_t, 0]
        h_ref_s = ref_np[mask_t, 0]

        t_label = f"{t_s:.0f}s"
        snapshots_error.append((x_s, y_s, err, t_label))
        snapshots_depth_pred.append((x_s, y_s, h_pred_s))
        snapshots_depth_ref.append((x_s, y_s, h_ref_s))

    # P2.1 multi-timestep error map
    if snapshots_error:
        try:
            fig = plot_error_maps_multi(
                snapshots=snapshots_error,
                save_path=os.path.join(plots_dir, "P2_1_error_maps.png"),
            )
        except Exception as exc:
            print(f"  Warning: P2.1 failed: {exc}")

    # P2.2 depth map at mid-time
    mid_idx = len(snapshots_depth_pred) // 2
    if snapshots_depth_pred:
        try:
            x_m, y_m, hp_m = snapshots_depth_pred[mid_idx]
            _, _, hr_m = snapshots_depth_ref[mid_idx]
            t_mid_label = snapshots_error[mid_idx][3]
            fig = plot_depth_map(
                x=x_m, y=y_m,
                h_pred=hp_m, h_ref=hr_m,
                time_label=t_mid_label,
                save_path=os.path.join(plots_dir, "P2_2_depth_map.png"),
            )
        except Exception as exc:
            print(f"  Warning: P2.2 failed: {exc}")

    # P2.7 spatial error decomposition at mid-time
    if snapshots_depth_pred:
        try:
            x_m, y_m, _ = snapshots_depth_pred[mid_idx]
            _, _, hr_m = snapshots_depth_ref[mid_idx]
            pred_full_m = predictions[
                np.abs(t_all - float(ts[ts.shape[0] // 2])) < (t_final / 40.0)
            ]
            coords_m = coords[
                np.abs(t_all - float(ts[ts.shape[0] // 2])) < (t_final / 40.0)
            ]
            cats = classify_points(
                np.asarray(coords_m), hr_m, domain_bounds,
            )
            err_m = np.abs(
                np.asarray(predictions[
                    np.abs(t_all - float(ts[ts.shape[0] // 2])) < (t_final / 40.0),
                    0
                ]) - hr_m
            )
            fig = plot_error_decomposition(
                x=x_m, y=y_m,
                error_h=err_m,
                categories=cats,
                time_label=snapshots_error[mid_idx][3],
                save_path=os.path.join(plots_dir, "P2_7_error_decomp.png"),
            )
        except Exception as exc:
            print(f"  Warning: P2.7 failed: {exc}")


# ---------------------------------------------------------------------------
# Multi-precision comparison (P3.1)
# ---------------------------------------------------------------------------

def _generate_precision_comparison(precision_results: dict, plots_dir: str) -> None:
    """Generate P3.1 precision comparison bar chart.

    Args:
        precision_results: Dict mapping precision label to results dict,
            e.g. {'float64': {'nse_h': 0.99, 'training_time_s': 120.0}, ...}
    """
    from evaluation.plots.comparisons import plot_precision_comparison_bar

    precisions = list(precision_results.keys())
    nse_vals = [precision_results[p].get("nse_h", float("nan")) for p in precisions]
    train_times = [precision_results[p].get("training_time_s", float("nan"))
                   for p in precisions]

    try:
        fig = plot_precision_comparison_bar(
            precisions=precisions,
            nse_values=nse_vals,
            training_times=train_times,
            save_path=os.path.join(plots_dir, "P3_1_precision_comparison.png"),
        )
    except Exception as exc:
        print(f"  Warning: P3.1 failed: {exc}")


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, float):
        if obj != obj:
            return "NaN"
        if obj == float("inf"):
            return "Inf"
        if obj == float("-inf"):
            return "-Inf"
    return obj


def _write_report(results: dict, output_dir: str) -> None:
    # YAML
    with open(os.path.join(output_dir, "summary_metrics.yaml"), "w") as f:
        yaml.dump(_sanitize(results), f, default_flow_style=False, sort_keys=False)

    # Plain text
    lines = ["=" * 64,
             "  Experiment 1 Postprocessing Report",
             "=" * 64, ""]

    acc = results.get("accuracy", {})
    lines.append("--- Accuracy (A1-A4) ---")
    for var in ["h", "hu", "hv"]:
        lines.append(f"  {var}:")
        for m in ["nse", "rmse", "mae", "rel_l2"]:
            key = f"{m}_{var}"
            if key in acc:
                lines.append(f"    {m:10s}: {acc[key]:.6f}")

    vb = results.get("volume_balance", {})
    if vb:
        lines += ["", "--- Volume Balance (B1) ---",
                  f"  max_mass_error_pct  : {vb.get('max_mass_error_pct', 'N/A'):.4f}",
                  f"  final_mass_error_pct: {vb.get('final_mass_error_pct', 'N/A'):.4f}"]

    bc = results.get("boundary", {})
    if bc:
        lines += ["", "--- Boundary Metrics ---"]
        for k, v in bc.items():
            lines.append(f"  {k}: {v}")

    infer = results.get("inference_cost", {})
    if infer:
        lines += ["", "--- Inference Cost (D3) ---",
                  f"  elapsed_s           : {infer.get('elapsed_s', 0):.4f}",
                  f"  throughput (pts/s)  : {infer.get('throughput_pts_per_s', 0):.0f}"]

    lines += ["", "=" * 64]

    report_text = "\n".join(lines)
    print(report_text)
    with open(os.path.join(output_dir, "report.txt"), "w") as f:
        f.write(report_text)

    # Save raw predictions
    np.savez_compressed(
        os.path.join(output_dir, "predictions.npz"),
        **results.get("_arrays", {}),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_postprocess(
    config_path: str,
    checkpoint_path: str,
    output_dir: str = None,
    precision_results_spec: str = None,
    training_history_path: str = None,
    batch_size: int = 50_000,
    skip_conservation: bool = False,
    skip_plots: bool = False,
) -> dict:
    """Run the full Experiment 1 postprocessing pipeline.

    Args:
        config_path: Path to experiment YAML config.
        checkpoint_path: Path to checkpoint directory (contains model.pkl).
        output_dir: Output directory. Auto-generated if None.
        precision_results_spec: Comma-separated 'label=path' pairs for P3.1,
            e.g. 'float64=results/exp1_f64,float32=results/exp1_f32'.
        training_history_path: JSON file with training history for P1.3/P1.4.
        batch_size: Prediction batch size.
        skip_conservation: If True, skip C3 (outflow AD gradient).
        skip_plots: If True, skip all plot generation.

    Returns:
        Results dict.
    """
    cfg_dict = load_config(config_path)
    experiment_name = get_experiment_name(cfg_dict, "experiment_1")

    if output_dir is None:
        ckpt_name = Path(checkpoint_path).name
        output_dir = os.path.join("inference_output", experiment_name, ckpt_name)
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")

    print(f"Postprocessing: experiment={experiment_name}, checkpoint={checkpoint_path}")

    # 1. Reconstruct model
    from flax.core import FrozenDict
    cfg = FrozenDict(cfg_dict)
    model, _p0, _tk, _vk = init_model_from_config(cfg)

    # 2. Load checkpoint
    params, train_meta = load_checkpoint(checkpoint_path)
    if params is None:
        raise FileNotFoundError(f"No model.pkl found in {checkpoint_path}")

    # 3. Generate validation grid
    print("  Generating analytical validation grid...")
    coords, targets, xs, ys, ts = _build_validation_grid(cfg_dict)
    print(f"  Grid: {coords.shape[0]:,} points ({xs.shape[0]}x{ys.shape[0]}x{ts.shape[0]})")

    # 4. Predict (timed for D3)
    print("  Running inference...")
    t0_infer = time.perf_counter()
    predictions = _predict_batched(model, params, coords, batch_size=batch_size)
    jax.block_until_ready(predictions)
    elapsed_infer = time.perf_counter() - t0_infer
    print(f"  Prediction: {coords.shape[0]:,} points in {elapsed_infer:.3f}s")

    # 5. Accuracy metrics (A1-A4)
    print("  Computing accuracy metrics...")
    accuracy = _compute_accuracy(predictions, targets)

    # 6. Conservation (B1)
    print("  Computing volume balance...")
    from evaluation.metrics.conservation import compute_volume_balance as _vb_fn
    vb_result = _vb_fn(
        np.asarray(predictions[:, 0]),
        None,  # no explicit cell areas
        None,  # no inflow volume
        None,  # no outflow volume
        domain_bounds={"lx": cfg_dict["domain"]["lx"],
                       "ly": cfg_dict["domain"]["ly"],
                       "t_final": cfg_dict["domain"]["t_final"]},
        coords=np.asarray(coords),
    )

    # 7. Boundary metrics (C1-C4)
    print("  Computing boundary metrics...")
    bpts = _build_boundary_points(cfg_dict, n_pts=500)
    boundary = _compute_boundary_metrics(
        model, params, bpts, cfg_dict, skip_conservation=skip_conservation
    )

    # 8. Inference cost (D3)
    infer_cost = {
        "elapsed_s": elapsed_infer,
        "n_points": int(coords.shape[0]),
        "throughput_pts_per_s": coords.shape[0] / elapsed_infer if elapsed_infer > 0 else float("inf"),
    }

    # Training cost from metadata (D2)
    training_cost_s = None
    if train_meta and "training_time_s" in train_meta:
        training_cost_s = train_meta["training_time_s"]

    # 9. Load optional training history
    training_history = None
    if training_history_path and os.path.exists(training_history_path):
        with open(training_history_path) as f:
            training_history = json.load(f)
        print(f"  Loaded training history from {training_history_path}")

    # 10. Assemble results
    results = {
        "experiment": experiment_name,
        "checkpoint": checkpoint_path,
        "n_points": int(coords.shape[0]),
        "accuracy": accuracy,
        "volume_balance": vb_result,
        "boundary": boundary,
        "inference_cost": infer_cost,
    }
    if training_cost_s is not None:
        results["training_cost_s"] = training_cost_s
    if train_meta:
        results["training_metadata"] = train_meta

    # Arrays for npz
    results["_arrays"] = {
        "coords": np.asarray(coords),
        "predictions": np.asarray(predictions),
        "targets": np.asarray(targets),
    }

    # 11. Write report
    _write_report(results, output_dir)
    print(f"  Summary saved to {output_dir}")

    # 12. Generate plots
    _generate_plots(
        coords, predictions, targets, ts, cfg_dict,
        plots_dir, training_history, skip_plots,
    )
    if not skip_plots:
        print(f"  Plots saved to {plots_dir}")

    # 13. P3.1 precision comparison (optional)
    if precision_results_spec:
        prec_results = {}
        for item in precision_results_spec.split(","):
            label, path = item.strip().split("=", 1)
            summary_path = os.path.join(path, "summary_metrics.yaml")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    prec_data = yaml.safe_load(f)
                acc_prec = prec_data.get("accuracy", {})
                meta_prec = prec_data.get("training_metadata", {})
                prec_results[label] = {
                    "nse_h": acc_prec.get("nse_h", float("nan")),
                    "training_time_s": meta_prec.get("training_time_s", float("nan")),
                }
        if prec_results:
            os.makedirs(plots_dir, exist_ok=True)
            _generate_precision_comparison(prec_results, plots_dir)

    # Remove internal arrays from returned dict
    results.pop("_arrays", None)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1 postprocessing pipeline (analytical dam-break)."
    )
    parser.add_argument("--config", required=True,
                        help="Path to experiment YAML config.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint directory.")
    parser.add_argument("--output", default=None,
                        help="Output directory (auto-generated if omitted).")
    parser.add_argument("--precision-results", default=None, dest="precision_results",
                        help="Comma-separated 'label=path' pairs for P3.1 precision comparison.")
    parser.add_argument("--training-history", default=None, dest="training_history",
                        help="Path to training_history.json for P1.3/P1.4 plots.")
    parser.add_argument("--batch-size", type=int, default=50_000, dest="batch_size",
                        help="Prediction batch size.")
    parser.add_argument("--skip-conservation", action="store_true", dest="skip_conservation",
                        help="Skip C3 (AD outflow gradient residual).")
    parser.add_argument("--skip-plots", action="store_true", dest="skip_plots",
                        help="Skip all plot generation.")
    args = parser.parse_args()

    # Ensure project root is on path
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    run_postprocess(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        precision_results_spec=args.precision_results,
        training_history_path=args.training_history,
        batch_size=args.batch_size,
        skip_conservation=args.skip_conservation,
        skip_plots=args.skip_plots,
    )


if __name__ == "__main__":
    main()
