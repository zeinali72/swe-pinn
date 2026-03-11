"""
L-BFGS fine-tuning for SWE PINN (Analytical Scenario) on a FIXED dataset.

Key changes vs naive implementation:
- Avoid reverse-mode AD through lax.scan over all batches (which can OOM).
- Instead, compute value+grad PER BATCH and accumulate across batches (memory-safe).
- Keep a deterministic value_fn for Optax line search (required by lbfgs solver).

Run from repo root, e.g.:
  python -m src.scenarios.analytical.analytical_lbfgs_finetune --config ... --weights_dir ...
"""

import os
import sys

# --- Ensure repo root is on sys.path BEFORE importing src.* ---
# This file is: <repo>/src/scenarios/analytical/analytical_lbfgs_finetune.py
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
    print(f"Added project root to path: {_REPO_ROOT}")

import time
import copy
import argparse
import importlib
import pickle
from typing import Any, Dict, Tuple, Optional, List

import jax
import jax.numpy as jnp
from jax import random, lax
import optax
from flax.core import FrozenDict

from src.config import load_config, DTYPE
from src.data import sample_domain, get_batches_tensor, get_sample_count
from src.models import init_model
from src.losses import (
    compute_pde_loss, compute_ic_loss, compute_bc_loss, total_loss,
    compute_data_loss, compute_neg_h_loss
)
from src.utils import nse, rmse
from src.physics import h_exact
from src.reporting import print_epoch_stats


def _is_pkl(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith(".pkl")


def _find_latest_pkl(weights_dir_or_file: str) -> str:
    if _is_pkl(weights_dir_or_file):
        return weights_dir_or_file
    if not os.path.isdir(weights_dir_or_file):
        raise FileNotFoundError(f"--weights_dir is neither a .pkl nor a directory: {weights_dir_or_file}")
    pkls = [os.path.join(weights_dir_or_file, fn) for fn in os.listdir(weights_dir_or_file) if fn.lower().endswith(".pkl")]
    if not pkls:
        raise FileNotFoundError(f"No .pkl files found in directory: {weights_dir_or_file}")
    pkls.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return pkls[0]


def _load_params_pkl(path: str) -> FrozenDict:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, FrozenDict):
        return obj
    if isinstance(obj, dict):
        return FrozenDict(obj)
    raise TypeError(f"Unsupported object type in pkl: {type(obj)}")


def _make_active_weights(cfg: FrozenDict, data_free: bool) -> Tuple[Dict[str, float], List[str]]:
    static_weights_dict = {k.replace("_weight", ""): v for k, v in cfg["loss_weights"].items()}
    active_loss_term_keys: List[str] = []
    for k, v in static_weights_dict.items():
        if v > 0:
            if k == "data" and data_free:
                continue
            if k == "building_bc":
                continue
            active_loss_term_keys.append(k)
    current_weights_dict = {k: static_weights_dict[k] for k in active_loss_term_keys}
    return current_weights_dict, active_loss_term_keys


def _compute_total_and_terms(
    model: Any,
    params: FrozenDict,
    all_batches: Dict[str, Any],
    weights_dict: Dict[str, float],
    config: FrozenDict,
    data_free: bool,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    active_loss_keys_base = list(weights_dict.keys())
    terms: Dict[str, jnp.ndarray] = {}

    # PDE (+ optional neg_h)
    pde_batch_data = all_batches.get("pde", jnp.empty((0, 3), dtype=DTYPE))
    if "pde" in active_loss_keys_base and pde_batch_data.shape[0] > 0:
        terms["pde"] = compute_pde_loss(model, params, pde_batch_data, config)
        if "neg_h" in active_loss_keys_base:
            terms["neg_h"] = compute_neg_h_loss(model, params, pde_batch_data)

    # IC
    ic_batch_data = all_batches.get("ic", jnp.empty((0, 3), dtype=DTYPE))
    if "ic" in active_loss_keys_base and ic_batch_data.shape[0] > 0:
        terms["ic"] = compute_ic_loss(model, params, ic_batch_data)

    # BC
    bc_batches = all_batches.get("bc", {})
    if "bc" in active_loss_keys_base and any(
        b.shape[0] > 0 for b in bc_batches.values() if hasattr(b, "shape") and b.shape[0] > 0
    ):
        terms["bc"] = compute_bc_loss(
            model,
            params,
            bc_batches.get("left", jnp.empty((0, 3), dtype=DTYPE)),
            bc_batches.get("right", jnp.empty((0, 3), dtype=DTYPE)),
            bc_batches.get("bottom", jnp.empty((0, 3), dtype=DTYPE)),
            bc_batches.get("top", jnp.empty((0, 3), dtype=DTYPE)),
            config,
        )

    # Data
    data_batch_data = all_batches.get("data", jnp.empty((0, 6), dtype=DTYPE))
    if (not data_free) and ("data" in active_loss_keys_base) and data_batch_data.shape[0] > 0:
        terms["data"] = compute_data_loss(model, params, data_batch_data, config)

    # Stable pytree keys
    terms_with_defaults = {k: terms.get(k, jnp.array(0.0, dtype=DTYPE)) for k in weights_dict.keys()}
    total = total_loss(terms_with_defaults, weights_dict)
    return total, terms_with_defaults


def _build_fixed_dataset(
    cfg: FrozenDict,
    data_free: bool,
    active_loss_term_keys: List[str],
    batch_size: int,
    fixed_key: jax.Array,
    data_points_full: Optional[jax.Array],
    max_batches: Optional[int],
) -> Tuple[Dict[str, Any], int]:
    sampling_cfg = cfg["sampling"]
    domain_cfg = cfg["domain"]

    n_pde = get_sample_count(sampling_cfg, "n_points_pde", 1000) if ("pde" in active_loss_term_keys or "neg_h" in active_loss_term_keys) else 0
    n_ic = get_sample_count(sampling_cfg, "n_points_ic", 100) if "ic" in active_loss_term_keys else 0
    n_bc_domain = get_sample_count(sampling_cfg, "n_points_bc_domain", 100) if "bc" in active_loss_term_keys else 0
    n_bc_per_wall = max(5, n_bc_domain // 4) if n_bc_domain > 0 else 0

    counts = [
        n_pde // batch_size,
        n_ic // batch_size,
        n_bc_per_wall // batch_size,
        n_bc_per_wall // batch_size,
        n_bc_per_wall // batch_size,
        n_bc_per_wall // batch_size,
    ]
    if (not data_free) and (data_points_full is not None):
        counts.append(int(data_points_full.shape[0]) // batch_size)

    num_batches = max(counts) if counts else 0
    if num_batches <= 0:
        raise ValueError(f"Computed num_batches={num_batches}. Batch size too large or too few points.")

    if max_batches is not None:
        num_batches = int(min(num_batches, max_batches))

    def generate_fixed_data(key):
        key, pde_key, ic_key, bc_keys, data_key = random.split(key, 5)

        # PDE
        if n_pde // batch_size > 0:
            pde_points = sample_domain(pde_key, n_pde, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"]))
            pde_data = get_batches_tensor(pde_key, pde_points, batch_size, num_batches)
        else:
            pde_data = jnp.zeros((num_batches, 0, 3), dtype=DTYPE)

        # IC
        if n_ic // batch_size > 0:
            ic_points = sample_domain(ic_key, n_ic, (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., 0.))
            ic_data = get_batches_tensor(ic_key, ic_points, batch_size, num_batches)
        else:
            ic_data = jnp.zeros((num_batches, 0, 3), dtype=DTYPE)

        # BC
        bc_data: Dict[str, jax.Array] = {}
        l_key, r_key, b_key, t_key = random.split(bc_keys, 4)

        def get_wall_data(k, n, x_rng, y_rng, t_rng):
            if n // batch_size > 0:
                pts = sample_domain(k, n, x_rng, y_rng, t_rng)
                return get_batches_tensor(k, pts, batch_size, num_batches)
            return jnp.zeros((num_batches, 0, 3), dtype=DTYPE)

        bc_data["left"] = get_wall_data(l_key, n_bc_per_wall, (0., 0.), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"]))
        bc_data["right"] = get_wall_data(r_key, n_bc_per_wall, (domain_cfg["lx"], domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"]))
        bc_data["bottom"] = get_wall_data(b_key, n_bc_per_wall, (0., domain_cfg["lx"]), (0., 0.), (0., domain_cfg["t_final"]))
        bc_data["top"] = get_wall_data(t_key, n_bc_per_wall, (0., domain_cfg["lx"]), (domain_cfg["ly"], domain_cfg["ly"]), (0., domain_cfg["t_final"]))

        # Data
        data_data = jnp.zeros((num_batches, 0, 6), dtype=DTYPE)
        if (not data_free) and (data_points_full is not None):
            data_data = get_batches_tensor(data_key, data_points_full, batch_size, num_batches)

        return {"pde": pde_data, "ic": ic_data, "bc": bc_data, "data": data_data, "building_bc": {}}

    fixed_batches = jax.jit(generate_fixed_data)(fixed_key)
    return fixed_batches, num_batches


def _make_value_fn_and_eval_terms(
    model: Any,
    cfg: FrozenDict,
    fixed_batches: Dict[str, Any],
    weights_dict: Dict[str, float],
    data_free: bool,
    num_batches: int,
):
    def eval_over_all_batches(params: FrozenDict):
        init_total = jnp.array(0.0, dtype=DTYPE)
        init_terms = {k: jnp.array(0.0, dtype=DTYPE) for k in weights_dict.keys()}

        def body(carry, batch_data):
            sum_total, sum_terms = carry
            all_batches = {
                "pde": batch_data["pde"],
                "ic": batch_data["ic"],
                "bc": batch_data["bc"],
                "data": batch_data["data"],
                "building_bc": batch_data["building_bc"],
            }
            total, terms = _compute_total_and_terms(model, params, all_batches, weights_dict, cfg, data_free)
            sum_total = sum_total + total
            sum_terms = {k: sum_terms[k] + terms[k] for k in sum_terms.keys()}
            return (sum_total, sum_terms), None

        (sum_total, sum_terms), _ = lax.scan(body, (init_total, init_terms), fixed_batches)
        mean_total = sum_total / jnp.array(float(num_batches), dtype=DTYPE)
        mean_terms = {k: sum_terms[k] / jnp.array(float(num_batches), dtype=DTYPE) for k in sum_terms.keys()}
        return mean_total, mean_terms

    eval_over_all_batches_jit = jax.jit(eval_over_all_batches)

    def value_fn(params: FrozenDict) -> jnp.ndarray:
        mean_total, _ = eval_over_all_batches_jit(params)
        return mean_total

    def eval_terms(params: FrozenDict) -> Tuple[float, Dict[str, float]]:
        mean_total, mean_terms = eval_over_all_batches_jit(params)
        return float(mean_total), {k: float(v) for k, v in mean_terms.items()}

    return value_fn, eval_terms


def _make_value_and_grad_accum(
    model: Any,
    cfg: FrozenDict,
    fixed_batches: Dict[str, Any],
    weights_dict: Dict[str, float],
    data_free: bool,
    num_batches: int,
):
    # Per-batch loss (scalar)
    def batch_loss(params: FrozenDict, batch_data: Dict[str, Any]) -> jnp.ndarray:
        all_batches = {
            "pde": batch_data["pde"],
            "ic": batch_data["ic"],
            "bc": batch_data["bc"],
            "data": batch_data["data"],
            "building_bc": batch_data["building_bc"],
        }
        total, _ = _compute_total_and_terms(model, params, all_batches, weights_dict, cfg, data_free)
        return total

    batch_vg = jax.value_and_grad(batch_loss)

    def value_and_grad_all_batches(params: FrozenDict):
        val0 = jnp.array(0.0, dtype=DTYPE)
        grad0 = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params)

        def body(carry, batch_data):
            val_sum, grad_sum = carry
            v, g = batch_vg(params, batch_data)  # reverse-mode only for ONE batch
            val_sum = val_sum + v
            grad_sum = jax.tree_util.tree_map(lambda a, b: a + b, grad_sum, g)
            return (val_sum, grad_sum), None

        (val_sum, grad_sum), _ = lax.scan(body, (val0, grad0), fixed_batches)

        mean_val = val_sum / jnp.array(float(num_batches), dtype=DTYPE)
        mean_grad = jax.tree_util.tree_map(lambda x: x / jnp.array(float(num_batches), dtype=DTYPE), grad_sum)
        return mean_val, mean_grad

    return jax.jit(value_and_grad_all_batches)


def main(weights_dir: str, config_path: str, out_path: str, max_iter: int, print_every: int, tol: float, max_batches: Optional[int]):
    weights_path = _find_latest_pkl(weights_dir)

    cfg_dict = load_config(config_path)
    cfg = FrozenDict(cfg_dict)

    # Init model (structure)
    models_module = importlib.import_module("src.models")
    model_class = getattr(models_module, cfg["model"]["name"])
    key = random.PRNGKey(int(cfg["training"]["seed"]))
    model_key, train_key, val_key, fixed_key = random.split(key, 4)
    model, _ = init_model(model_class, model_key, cfg)

    # Load params
    params = _load_params_pkl(weights_path)

    # data_free flag
    data_free_flag = cfg.get("data_free")
    data_free = (data_free_flag is not False)
    if data_free:
        print("Info: 'data_free: true' in config. Fine-tuning in data-free mode.")
    else:
        print("Info: 'data_free: false' in config. Fine-tuning in data-driven mode (if train_grid is present).")

    # (Optional) build analytical data_points_full if data-driven
    data_points_full = None
    if not data_free:
        try:
            train_grid_cfg = cfg["train_grid"]
            domain_cfg = cfg["domain"]
            data_points_coords = sample_domain(
                train_key, train_grid_cfg["n_points_train"],
                (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])
            )
            h_true_train = h_exact(
                data_points_coords[:, 0], data_points_coords[:, 2],
                cfg["physics"]["n_manning"], cfg["physics"]["u_const"]
            )
            u_true_train = jnp.full_like(h_true_train, cfg["physics"]["u_const"])
            v_true_train = jnp.zeros_like(h_true_train)
            data_points_full = jnp.stack([
                data_points_coords[:, 2], data_points_coords[:, 0], data_points_coords[:, 1],
                h_true_train, u_true_train, v_true_train
            ], axis=1).astype(DTYPE)
            if data_points_full.shape[0] == 0:
                data_points_full = None
                data_free = True
        except Exception:
            data_points_full = None
            data_free = True

    current_weights_dict, active_loss_term_keys = _make_active_weights(cfg, data_free=data_free)

    print(f"Loaded weights from: {weights_path}")
    print(f"Config: {config_path}")
    print(f"Active Loss Terms: {active_loss_term_keys}")
    print(f"Using weights: {current_weights_dict}")

    # Validation set (analytical)
    validation_data_loaded = False
    val_points, h_true_val = None, None
    try:
        val_grid_cfg = cfg["validation_grid"]
        domain_cfg = cfg["domain"]
        val_points = sample_domain(
            val_key, val_grid_cfg["n_points_val"],
            (0., domain_cfg["lx"]), (0., domain_cfg["ly"]), (0., domain_cfg["t_final"])
        )
        h_true_val = h_exact(val_points[:, 0], val_points[:, 2], cfg["physics"]["n_manning"], cfg["physics"]["u_const"])
        validation_data_loaded = (val_points.shape[0] > 0)
        if validation_data_loaded:
            print(f"Validation set: {val_points.shape[0]} points.")
    except Exception:
        print("Warning: validation_grid missing/invalid; NSE/RMSE disabled.")

    # Fixed dataset
    batch_size = int(cfg["training"]["batch_size"])
    fixed_batches, num_batches = _build_fixed_dataset(
        cfg=cfg,
        data_free=data_free,
        active_loss_term_keys=active_loss_term_keys,
        batch_size=batch_size,
        fixed_key=fixed_key,
        data_points_full=data_points_full,
        max_batches=max_batches,
    )
    print(f"Fixed collocation dataset built: num_batches={num_batches}, batch_size={batch_size}")

    # value_fn + terms + memory-safe value_and_grad
    value_fn, eval_terms = _make_value_fn_and_eval_terms(
        model=model,
        cfg=cfg,
        fixed_batches=fixed_batches,
        weights_dict=current_weights_dict,
        data_free=data_free,
        num_batches=num_batches,
    )
    value_and_grad_all = _make_value_and_grad_accum(
        model=model,
        cfg=cfg,
        fixed_batches=fixed_batches,
        weights_dict=current_weights_dict,
        data_free=data_free,
        num_batches=num_batches,
    )

    # L-BFGS solver (uses line search via extra args)
    solver = optax.lbfgs()

    state = solver.init(params)

    print("\n--- L-BFGS Fine-tuning Started ---")
    start_time = time.time()
    last_time = time.time()

    best_nse = -jnp.inf
    best_params = copy.deepcopy(params)

    for it in range(1, max_iter + 1):
        # Memory-safe: batchwise VJP + accumulate
        value, grad = value_and_grad_all(params)

        # Optax line-search update requires value, grad, and the function itself (value_fn) :contentReference[oaicite:1]{index=1}
        updates, state = solver.update(grad, state, params, value=value, grad=grad, value_fn=value_fn)
        params = optax.apply_updates(params, updates)

        grad_norm = float(optax.tree_utils.tree_l2_norm(grad))

        if (it % print_every) == 0 or it == 1:
            mean_total, mean_terms = eval_terms(params)

            nse_val, rmse_val = -jnp.inf, jnp.inf
            if validation_data_loaded:
                U_pred_val = model.apply({"params": params["params"]}, val_points, train=False)
                h_pred_val = U_pred_val[..., 0]
                nse_val = float(nse(h_pred_val, h_true_val))
                rmse_val = float(rmse(h_pred_val, h_true_val))
                if nse_val > best_nse:
                    best_nse = nse_val
                    best_params = copy.deepcopy(params)
                    print(f"    ---> New best NSE: {best_nse:.6f} at iteration {it}")

            epoch_time = time.time() - last_time
            last_time = time.time()

            print_epoch_stats(
                it, it, start_time, mean_total,
                mean_terms.get("pde", 0.0),
                mean_terms.get("ic", 0.0),
                mean_terms.get("bc", 0.0),
                0.0,  # building_bc_loss
                mean_terms.get("data", 0.0),
                mean_terms.get("neg_h", 0.0),
                nse_val, rmse_val, epoch_time
            )
            print(f"    grad_norm: {grad_norm:.3e}")

        if tol > 0 and grad_norm < tol:
            print(f"Stopping early: grad_norm {grad_norm:.3e} < tol {tol:.3e}")
            break

    # Save
    params_to_save = best_params if validation_data_loaded else params
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(params_to_save, f)

    print("\n--- L-BFGS Fine-tuning Finished ---")
    if validation_data_loaded and best_nse > -jnp.inf:
        print(f"Best NSE achieved during L-BFGS: {best_nse:.6f}")
    print(f"Saved fine-tuned params to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L-BFGS fine-tuning for SWE PINN (Analytical Scenario).")
    parser.add_argument("--weights_dir", type=str, required=True,
                        help="Path to a .pkl file OR a directory containing .pkl checkpoints.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML config used to build the model.")
    parser.add_argument("--out", type=str, required=True,
                        help="Output .pkl path for fine-tuned params.")
    parser.add_argument("--max_iter", type=int, default=200, help="Maximum L-BFGS iterations.")
    parser.add_argument("--print_every", type=int, default=10, help="Print metrics every N iterations.")
    parser.add_argument("--tol", type=float, default=0.0,
                        help="Optional early-stop tolerance on grad norm (0 disables). Example: 1e-6")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Optional cap on number of fixed batches (reduces compute; also safer for memory). Example: 50")
    args = parser.parse_args()

    main(
        weights_dir=args.weights_dir,
        config_path=args.config,
        out_path=args.out,
        max_iter=args.max_iter,
        print_every=args.print_every,
        tol=args.tol,
        max_batches=args.max_batches,
    )
