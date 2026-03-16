"""Report generation: YAML summary, CSV tables, text report, raw predictions."""
import csv
import os
from typing import Dict

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# YAML summary
# ---------------------------------------------------------------------------

def _sanitize_for_yaml(obj):
    """Recursively convert numpy/jax types to plain Python for YAML."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_yaml(v) for v in obj]
    if hasattr(obj, "item"):  # numpy/jax scalar
        return obj.item()
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return "NaN"
        if obj == float("inf"):
            return "Inf"
        if obj == float("-inf"):
            return "-Inf"
    return obj


def save_yaml_summary(results: dict, output_dir: str) -> str:
    """Write ``summary_metrics.yaml`` to *output_dir*."""
    path = os.path.join(output_dir, "summary_metrics.yaml")
    with open(path, "w") as f:
        yaml.dump(_sanitize_for_yaml(results), f, default_flow_style=False, sort_keys=False)
    return path


# ---------------------------------------------------------------------------
# CSV tables
# ---------------------------------------------------------------------------

def save_csv_tables(results: dict, output_dir: str):
    """Write spatial and temporal decomposition tables as CSV."""
    # Spatial decomposition
    spatial = results.get("spatial_decomposition")
    if spatial:
        path = os.path.join(output_dir, "spatial_decomposition.csv")
        _dict_of_dicts_to_csv(spatial, path, index_name="region")

    # Temporal decomposition
    temporal = results.get("temporal_decomposition")
    if temporal:
        path = os.path.join(output_dir, "temporal_decomposition.csv")
        rows = []
        for phase, metrics in temporal.items():
            row = {"phase": phase}
            for k, v in metrics.items():
                if isinstance(v, dict):
                    for mk, mv in v.items():
                        row[f"{k}_{mk}"] = mv
                else:
                    row[k] = v
            rows.append(row)
        if rows:
            fieldnames = list(rows[0].keys())
            for r in rows[1:]:
                for k in r:
                    if k not in fieldnames:
                        fieldnames.append(k)
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)


def _dict_of_dicts_to_csv(data: dict, path: str, index_name: str = "key"):
    """Write a dict-of-dicts as a CSV with one row per outer key."""
    rows = []
    for key, inner in data.items():
        row = {index_name: key}
        if isinstance(inner, dict):
            row.update(inner)
        rows.append(row)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Raw predictions
# ---------------------------------------------------------------------------

def save_raw_predictions(ctx, output_dir: str) -> str:
    """Save coordinates, predictions, and targets as ``predictions.npz``."""
    path = os.path.join(output_dir, "predictions.npz")
    np.savez_compressed(
        path,
        coords=np.asarray(ctx.val_coords),
        predictions=np.asarray(ctx.predictions),
        targets=np.asarray(ctx.val_targets),
    )
    return path


# ---------------------------------------------------------------------------
# Text report (console + file)
# ---------------------------------------------------------------------------

def print_text_report(results: dict, output_dir: str = None):
    """Print a formatted summary to stdout and optionally save as report.txt."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  Inference Report  —  checkpoint: {results.get('checkpoint', '?')}")
    lines.append("=" * 60)
    lines.append(f"  Points evaluated : {results.get('n_points', '?'):,}")
    lines.append(f"  Inference time   : {results.get('inference_time_seconds', 0):.3f} s")
    lines.append("")

    # Accuracy
    acc = results.get("accuracy", {})
    lines.append("--- Accuracy Metrics ---")
    for key in ["nse_h", "rmse_h", "mae_h", "rel_l2_h", "r_squared_h",
                "nse_hu", "rmse_hu", "nse_hv", "rmse_hv",
                "peak_depth_error", "time_to_peak_error", "rmse_mae_ratio_h"]:
        if key in acc:
            lines.append(f"  {key:30s} : {acc[key]:.6f}")

    neg = acc.get("negative_depth", {})
    if neg:
        lines.append(f"  {'neg_depth_fraction':30s} : {neg.get('fraction', 0):.6f}")
        lines.append(f"  {'neg_depth_min_h':30s} : {neg.get('min_h', 0):.6f}")

    # Conservation
    vb = results.get("volume_balance")
    if vb:
        lines.append("")
        lines.append("--- Volume Balance ---")
        lines.append(f"  max_mass_error_pct           : {vb['max_mass_error_pct']:.4f}")
        lines.append(f"  final_mass_error_pct         : {vb['final_mass_error_pct']:.4f}")

    cr = results.get("continuity_residual")
    if cr:
        lines.append("")
        lines.append("--- Continuity Residual ---")
        lines.append(f"  mean_abs                     : {cr['mean_abs']:.6e}")
        lines.append(f"  max_abs                      : {cr['max_abs']:.6e}")

    # Flood extent
    fe = results.get("flood_extent")
    if fe:
        lines.append("")
        lines.append("--- Flood Extent ---")
        for th_key, vals in fe.items():
            lines.append(f"  {th_key}: CSI={vals['csi']:.4f}  HR={vals['hit_rate']:.4f}  FAR={vals['far']:.4f}")

    # IC accuracy
    ic = results.get("initial_condition")
    if ic:
        lines.append("")
        lines.append("--- Initial Condition ---")
        lines.append(f"  rmse                         : {ic.get('rmse', 0):.6e}")
        lines.append(f"  max_abs_error                : {ic.get('max_abs_error', 0):.6e}")

    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    if output_dir:
        path = os.path.join(output_dir, "report.txt")
        with open(path, "w") as f:
            f.write(report)


# ---------------------------------------------------------------------------
# Multi-checkpoint comparison
# ---------------------------------------------------------------------------

def save_comparison_table(all_results: Dict[str, dict], output_dir: str):
    """Write a comparison CSV across multiple checkpoints."""
    rows = []
    for ckpt_name, results in all_results.items():
        row = {"checkpoint": ckpt_name}
        acc = results.get("accuracy", {})
        for key in ["nse_h", "rmse_h", "mae_h", "r_squared_h",
                    "peak_depth_error", "time_to_peak_error"]:
            row[key] = acc.get(key)
        neg = acc.get("negative_depth", {})
        row["neg_depth_fraction"] = neg.get("fraction")
        row["inference_time_s"] = results.get("inference_time_seconds")
        rows.append(row)

    if not rows:
        return

    path = os.path.join(output_dir, "checkpoint_comparison.csv")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Comparison table written to {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def generate_plots(ctx, results: dict, output_dir: str):
    """Generate standard comparison plots."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    targets = ctx.val_targets
    preds = ctx.predictions
    coords = ctx.val_coords

    try:
        from src.utils.plotting import plot_comparison_scatter_2d

        # Find a representative time slice
        plot_cfg = ctx.config.get("plotting", {})
        t_const = plot_cfg.get("t_const_val", float(ctx.domain_bounds.get("t_final", 1.0)) / 2)
        t_all = coords[:, 2]
        unique_t = jnp.unique(t_all)
        t_closest = unique_t[jnp.argmin(jnp.abs(unique_t - t_const))]
        t_mask = jnp.abs(t_all - t_closest) < 1e-6

        if jnp.sum(t_mask) > 3 and targets.shape[-1] >= 3:
            x_slice = coords[t_mask, 0]
            y_slice = coords[t_mask, 1]

            for i, var in enumerate(["h", "hu", "hv"]):
                filename = os.path.join(plots_dir, f"comparison_{var}.png")
                plot_comparison_scatter_2d(
                    x_slice, y_slice,
                    preds[t_mask, i], targets[t_mask, i],
                    var, ctx.config, filename=filename,
                )

        # 1D plot for experiment 1
        if ctx.experiment_meta.get("reference_type") == "analytical":
            from src.utils.plotting import plot_h_vs_x
            y_const = plot_cfg.get("y_const_plot", 0)
            y_mask = jnp.abs(coords[:, 1] - y_const) < 1e-6
            yt_mask = t_mask & y_mask
            if jnp.sum(yt_mask) > 1:
                x_line = coords[yt_mask, 0]
                sort_idx = jnp.argsort(x_line)
                plot_h_vs_x(
                    x_line[sort_idx],
                    preds[yt_mask, 0][sort_idx],
                    float(t_closest), float(y_const),
                    ctx.config,
                    filename=os.path.join(plots_dir, "h_vs_x.png"),
                )

        # Error heatmap for h
        if jnp.sum(t_mask) > 3 and targets.shape[-1] >= 1:
            _plot_error_heatmap(
                coords[t_mask, 0], coords[t_mask, 1],
                preds[t_mask, 0], targets[t_mask, 0],
                os.path.join(plots_dir, "error_h.png"),
            )

    except Exception as e:
        print(f"  Warning: Plot generation failed: {e}")


def _plot_error_heatmap(x, y, pred, true, filename):
    """Simple error heatmap using tricontourf."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    error = pred - true
    try:
        triang = mtri.Triangulation(x, y)
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    v_max = max(float(jnp.max(jnp.abs(error))), 1e-9)
    levels = jnp.linspace(-v_max, v_max, 51)
    cf = ax.tricontourf(triang, error, levels=levels, cmap="coolwarm", extend="both")
    fig.colorbar(cf, ax=ax, label="Error (pred - true)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Water depth error")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# Lazy import for jnp used in generate_plots
import jax.numpy as jnp
