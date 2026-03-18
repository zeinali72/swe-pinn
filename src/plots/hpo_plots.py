"""HPO diagnostic plots: P4.1-P4.5."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from typing import Dict, List, Optional

# Colour palette
EXETER_DEEP_GREEN = "#003C3C"
EXETER_TEAL = "#007D69"
EXETER_MINT = "#00C896"
BLUE_HEART_NAVY = "#0D2B45"
BLUE_HEART_OCEAN = "#1B5E8A"
BLUE_HEART_SKY = "#4FA3D1"


def _apply_style() -> None:
    """Apply project-standard matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans"],
            "figure.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_optimisation_history(
    trial_numbers: np.ndarray,
    nse_values: np.ndarray,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """P4.1 — HPO optimisation history scatter with running best.

    Parameters
    ----------
    trial_numbers:
        1-D array of trial indices.
    nse_values:
        1-D array of NSE values (primary objective) per trial.
    save_path:
        If provided, saved here at 300 DPI.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()

    if trial_numbers is None or len(trial_numbers) == 0:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, "No data provided", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("HPO Optimisation History")
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return fig

    trial_numbers = np.asarray(trial_numbers)
    nse_values = np.asarray(nse_values, dtype=float)

    # Compute running best (maximising NSE)
    running_best = np.maximum.accumulate(np.where(np.isnan(nse_values), -np.inf, nse_values))
    is_best = np.zeros(len(nse_values), dtype=bool)
    for i, val in enumerate(nse_values):
        if not np.isnan(val) and (i == 0 or val >= running_best[i - 1]):
            is_best[i] = True

    fig, ax = plt.subplots(figsize=(10, 5))

    # All trials
    ax.scatter(
        trial_numbers[~is_best],
        nse_values[~is_best],
        color=BLUE_HEART_OCEAN,
        s=30,
        alpha=0.6,
        label="Trial",
        zorder=3,
        marker="o",
    )

    # Best trials
    ax.scatter(
        trial_numbers[is_best],
        nse_values[is_best],
        color=EXETER_MINT,
        s=80,
        alpha=0.9,
        label="New best",
        zorder=5,
        marker="*",
    )

    # Running best line
    valid_mask = ~np.isneginf(running_best)
    ax.step(
        trial_numbers[valid_mask],
        running_best[valid_mask],
        where="post",
        color=EXETER_DEEP_GREEN,
        linewidth=1.5,
        linestyle="--",
        label="Running best",
        zorder=4,
    )

    ax.set_xlabel("Trial number")
    ax.set_ylabel("NSE (h)")
    ax.set_title("HPO Optimisation History")
    ax.legend(frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_parallel_coordinates(
    trials_df,
    hp_columns: List[str],
    nse_column: str,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """P4.2 — Parallel coordinates plot for HPO trials.

    Lines are coloured by NSE value (low → blue, high → green).

    Parameters
    ----------
    trials_df:
        ``pandas.DataFrame`` (or array-like) with one row per trial and
        columns named by *hp_columns* plus *nse_column*.
    hp_columns:
        Ordered list of hyperparameter column names to plot as axes.
    nse_column:
        Name of the NSE column used to colour the lines.
    save_path:
        If provided, saved here at 300 DPI.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()

    # Accept pandas DataFrame or structured numpy array
    try:
        import pandas as pd
        if not isinstance(trials_df, pd.DataFrame):
            trials_df = pd.DataFrame(trials_df)
        all_cols = hp_columns + [nse_column]
        df = trials_df[all_cols].dropna()
    except ImportError:
        # Fallback: assume trials_df is already a 2-D numpy array with
        # columns corresponding to hp_columns + [nse_column]
        import numpy as _np
        arr = _np.asarray(trials_df, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != len(hp_columns) + 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.text(0.5, 0.5, "Invalid data shape", ha="center", va="center", transform=ax.transAxes)
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return fig

        class _SimpleDF:
            def __init__(self, arr, cols):
                self._arr = arr
                self._cols = cols

            def __getitem__(self, cols):
                idx = [self._cols.index(c) for c in cols]
                return self._arr[:, idx]

            def iterrows(self):
                for i, row in enumerate(self._arr):
                    yield i, {c: row[j] for j, c in enumerate(self._cols)}

            def __len__(self):
                return len(self._arr)

        all_cols = hp_columns + [nse_column]
        df = _SimpleDF(arr, all_cols)

    n_axes = len(hp_columns)

    if n_axes < 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.text(
            0.5, 0.5,
            "Need at least 2 hyperparameter axes",
            ha="center", va="center", transform=ax.transAxes,
        )
        ax.set_title("Parallel Coordinates")
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return fig

    fig, axes = plt.subplots(1, n_axes - 1, figsize=(max(10, 2 * n_axes), 6), sharey=False)
    if n_axes == 2:
        axes = [axes]

    # Normalise each column to [0, 1] for drawing
    try:
        import pandas as pd
        col_data = {col: df[col].values.astype(float) for col in hp_columns + [nse_column]}
    except Exception:
        # _SimpleDF path
        col_data = {}
        for i, col in enumerate(hp_columns + [nse_column]):
            col_data[col] = np.asarray(
                [row[1][col] for row in df.iterrows()], dtype=float
            )

    def _normalise(arr):
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        if hi == lo:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    normed = {col: _normalise(col_data[col]) for col in hp_columns + [nse_column]}

    nse_norm = normed[nse_column]
    cmap = cm.get_cmap("RdYlGn")
    n_trials = len(col_data[hp_columns[0]])

    for trial_idx in range(n_trials):
        colour = cmap(float(nse_norm[trial_idx]))
        for ax_idx, ax in enumerate(axes):
            y_left = normed[hp_columns[ax_idx]][trial_idx]
            y_right = normed[hp_columns[ax_idx + 1]][trial_idx]
            ax.plot([0, 1], [y_left, y_right], color=colour, alpha=0.4, linewidth=0.8)

    for ax_idx, ax in enumerate(axes):
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([hp_columns[ax_idx], hp_columns[ax_idx + 1]], fontsize=8)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    # Colorbar for NSE
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
        vmin=float(np.nanmin(col_data[nse_column])),
        vmax=float(np.nanmax(col_data[nse_column])),
    ))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_label(nse_column)

    fig.suptitle("Parallel Coordinates: Hyperparameter Trials", y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_hp_importance(
    importance_dict: Dict[str, float],
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """P4.3 — Horizontal bar chart of hyperparameter importance scores.

    Parameters
    ----------
    importance_dict:
        Mapping of hyperparameter name to importance score (e.g. from
        ``optuna.importance.get_param_importances``).
    save_path:
        If provided, saved here at 300 DPI.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _apply_style()

    if not importance_dict:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data provided", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Hyperparameter Importance")
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return fig

    # Sort by importance descending
    sorted_items = sorted(importance_dict.items(), key=lambda kv: kv[1], reverse=True)
    names = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(names) + 1)))

    y_pos = np.arange(len(names))
    ax.barh(y_pos, scores, color=EXETER_TEAL, alpha=0.85, height=0.6)

    for i, (score, name) in enumerate(zip(scores, names)):
        ax.text(
            score + max(scores) * 0.01,
            i,
            f"{score:.3f}",
            va="center",
            fontsize=8,
            color=EXETER_DEEP_GREEN,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance score")
    ax.set_title("Hyperparameter Importance")
    ax.invert_yaxis()  # Most important at top

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_hp_slice(*args, **kwargs):
    """P4.4 — Hyperparameter slice plot. Not yet implemented."""
    raise NotImplementedError("Not yet implemented")


def plot_hp_contour(*args, **kwargs):
    """P4.5 — Hyperparameter contour plot. Not yet implemented."""
    raise NotImplementedError("Not yet implemented")
