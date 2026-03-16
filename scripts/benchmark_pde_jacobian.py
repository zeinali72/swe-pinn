#!/usr/bin/env python3
"""Benchmark PDE Jacobian computation across architectures and batch sizes.

Usage:
    python scripts/benchmark_pde_jacobian.py --config configs/experiment_1.yaml
    python scripts/benchmark_pde_jacobian.py --config configs/experiment_1.yaml --output profiling/

Produces:
    - Console timing breakdown per component
    - Optional JSON report in output directory

Benchmarks:
    1. Jacobian wall-clock time at various batch sizes
    2. jacfwd vs jacrev comparison
    3. Per-architecture benchmark (MLP, FourierPINN, DGM)
    4. Full train_step breakdown by loss component
"""
import argparse
import json
import os
import time
import timeit

import jax
import jax.numpy as jnp
from jax import random
from flax.core import FrozenDict

from src.config import load_config, get_dtype, DTYPE
from src.models.pinn import MLP, FourierPINN, DGMNetwork
from src.models.factory import init_model


def _make_U_fn(model, params):
    """Create a closure for model forward pass (matches losses/pde.py)."""
    def U_fn(pts):
        return model.apply({'params': params['params']}, pts, train=False)
    return U_fn


def benchmark_jacobian_batch_sizes(model, params, key, dtype, batch_sizes=None):
    """Benchmark Jacobian computation at various batch sizes."""
    if batch_sizes is None:
        batch_sizes = [100, 500, 1000, 5000, 10000]

    U_fn = _make_U_fn(model, params)
    jac_fn = jax.jit(jax.vmap(jax.jacfwd(U_fn)))

    # Warmup (JIT compile)
    warmup_batch = jnp.ones((1, 3), dtype=dtype)
    _ = jac_fn(warmup_batch)
    jax.block_until_ready(_)

    print("\n=== Jacobian Wall-Clock Time vs Batch Size ===")
    results = {}
    for n in batch_sizes:
        batch = random.uniform(key, (n, 3), dtype=dtype)
        n_repeats = max(1, min(10, 50000 // n))
        t = timeit.timeit(lambda: jax.block_until_ready(jac_fn(batch)), number=n_repeats) / n_repeats
        us_per_point = t / n * 1e6
        print(f"  n={n:>6}: {t*1000:.1f} ms  ({us_per_point:.1f} us/point)")
        results[n] = {"total_ms": t * 1000, "us_per_point": us_per_point}

    return results


def benchmark_jacfwd_vs_jacrev(model, params, key, dtype, n_points=1000):
    """Compare jacfwd vs jacrev for the model."""
    U_fn = _make_U_fn(model, params)

    jac_fwd_fn = jax.jit(jax.vmap(jax.jacfwd(U_fn)))
    jac_rev_fn = jax.jit(jax.vmap(jax.jacrev(U_fn)))

    batch = random.uniform(key, (n_points, 3), dtype=dtype)

    # Warmup both
    _ = jac_fwd_fn(batch)
    jax.block_until_ready(_)
    _ = jac_rev_fn(batch)
    jax.block_until_ready(_)

    n_repeats = 10
    t_fwd = timeit.timeit(lambda: jax.block_until_ready(jac_fwd_fn(batch)), number=n_repeats) / n_repeats
    t_rev = timeit.timeit(lambda: jax.block_until_ready(jac_rev_fn(batch)), number=n_repeats) / n_repeats

    print(f"\n=== jacfwd vs jacrev (n={n_points}) ===")
    print(f"  jacfwd: {t_fwd*1000:.1f} ms  ({t_fwd/n_points*1e6:.1f} us/point)")
    print(f"  jacrev: {t_rev*1000:.1f} ms  ({t_rev/n_points*1e6:.1f} us/point)")
    ratio = t_fwd / max(t_rev, 1e-9)
    print(f"  Ratio (fwd/rev): {ratio:.2f}x")

    return {
        "n_points": n_points,
        "jacfwd_ms": t_fwd * 1000,
        "jacrev_ms": t_rev * 1000,
        "ratio_fwd_over_rev": ratio,
    }


def benchmark_architectures(cfg_dict, key, dtype):
    """Benchmark Jacobian computation across MLP, FourierPINN, and DGM."""
    # Ensure comparable configs
    base_model_cfg = {
        "width": cfg_dict["model"].get("width", 128),
        "depth": cfg_dict["model"].get("depth", 3),
        "output_dim": cfg_dict["model"].get("output_dim", 3),
        "bias_init": cfg_dict["model"].get("bias_init", 0.0),
    }

    arch_configs = {
        "MLP": {**base_model_cfg, "name": "MLP"},
        "FourierPINN": {
            **base_model_cfg, "name": "FourierPINN",
            "ff_dims": cfg_dict["model"].get("ff_dims", 64),
            "fourier_scale": cfg_dict["model"].get("fourier_scale", 1.0),
        },
        "DGM": {**base_model_cfg, "name": "DGMNetwork"},
    }

    n_points = 1000
    batch = random.uniform(key, (n_points, 3), dtype=dtype)

    print(f"\n=== Per-Architecture Benchmark (n={n_points}, width={base_model_cfg['width']}, depth={base_model_cfg['depth']}) ===")
    results = {}

    for arch_name, model_cfg in arch_configs.items():
        try:
            cfg_copy = dict(cfg_dict)
            cfg_copy["model"] = model_cfg

            model_class = {"MLP": MLP, "FourierPINN": FourierPINN, "DGMNetwork": DGMNetwork}[model_cfg["name"]]
            model, params = init_model(model_class, key, FrozenDict(cfg_copy))

            U_fn = _make_U_fn(model, params)
            jac_fn = jax.jit(jax.vmap(jax.jacfwd(U_fn)))

            # Warmup
            _ = jac_fn(batch[:1])
            jax.block_until_ready(_)

            n_repeats = 10
            t = timeit.timeit(lambda: jax.block_until_ready(jac_fn(batch)), number=n_repeats) / n_repeats

            print(f"  {arch_name:<15}: {t*1000:.1f} ms  ({t/n_points*1e6:.1f} us/point)")
            results[arch_name] = {"total_ms": t * 1000, "us_per_point": t / n_points * 1e6}
        except Exception as e:
            print(f"  {arch_name:<15}: FAILED — {e}")
            results[arch_name] = {"error": str(e)}

    return results


def benchmark_loss_components(model, params, key, cfg_dict, dtype, n_points=1000):
    """Break down train_step time by loss component."""
    from src.losses.pde import compute_pde_loss, compute_neg_h_loss, compute_ic_loss
    from src.losses.boundary import loss_boundary_neumann_outflow_x, loss_boundary_wall_horizontal

    config = FrozenDict(cfg_dict)
    batch = random.uniform(key, (n_points, 3), dtype=dtype)
    ic_batch = random.uniform(key, (n_points, 3), dtype=dtype)
    ic_batch = ic_batch.at[..., 2].set(0.0)  # t=0 for IC

    components = {}

    print(f"\n=== Loss Component Breakdown (n={n_points}) ===")

    # PDE loss (contains Jacobian)
    pde_jit = jax.jit(lambda p: compute_pde_loss(model, p, batch, config))
    _ = pde_jit(params)
    jax.block_until_ready(_)
    t = timeit.timeit(lambda: jax.block_until_ready(pde_jit(params)), number=10) / 10
    print(f"  PDE loss (with Jacobian):  {t*1000:.1f} ms")
    components["pde_loss"] = t * 1000

    # IC loss (forward only)
    ic_jit = jax.jit(lambda p: compute_ic_loss(model, p, ic_batch))
    _ = ic_jit(params)
    jax.block_until_ready(_)
    t = timeit.timeit(lambda: jax.block_until_ready(ic_jit(params)), number=10) / 10
    print(f"  IC loss (forward only):    {t*1000:.1f} ms")
    components["ic_loss"] = t * 1000

    # Neg h loss (forward only)
    neg_jit = jax.jit(lambda p: compute_neg_h_loss(model, p, batch))
    _ = neg_jit(params)
    jax.block_until_ready(_)
    t = timeit.timeit(lambda: jax.block_until_ready(neg_jit(params)), number=10) / 10
    print(f"  Neg-h loss (forward only): {t*1000:.1f} ms")
    components["neg_h_loss"] = t * 1000

    # Neumann BC loss (has Jacobian)
    bc_jit = jax.jit(lambda p: loss_boundary_neumann_outflow_x(model, p, batch))
    _ = bc_jit(params)
    jax.block_until_ready(_)
    t = timeit.timeit(lambda: jax.block_until_ready(bc_jit(params)), number=10) / 10
    print(f"  Neumann BC (with Jac):     {t*1000:.1f} ms")
    components["neumann_bc_loss"] = t * 1000

    # Wall BC loss (forward only)
    wall_jit = jax.jit(lambda p: loss_boundary_wall_horizontal(model, p, batch))
    _ = wall_jit(params)
    jax.block_until_ready(_)
    t = timeit.timeit(lambda: jax.block_until_ready(wall_jit(params)), number=10) / 10
    print(f"  Wall BC (forward only):    {t*1000:.1f} ms")
    components["wall_bc_loss"] = t * 1000

    # Full gradient of PDE loss (backward through Jacobian)
    grad_pde_jit = jax.jit(jax.grad(lambda p: compute_pde_loss(model, p, batch, config)))
    _ = grad_pde_jit(params)
    jax.block_until_ready(_)
    t = timeit.timeit(lambda: jax.block_until_ready(grad_pde_jit(params)), number=10) / 10
    print(f"  grad(PDE loss) [backward]: {t*1000:.1f} ms")
    components["grad_pde_loss"] = t * 1000

    return components


def main():
    parser = argparse.ArgumentParser(description="Benchmark PDE Jacobian computation")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", default=None, help="Output directory for JSON report")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    dtype = get_dtype(cfg_dict)
    key = random.PRNGKey(cfg_dict["training"].get("seed", 42))

    print(f"Config: {args.config}")
    print(f"Devices: {jax.devices()}")
    print(f"dtype: {dtype}")

    # Initialize the model from config
    model_name = cfg_dict["model"]["name"]
    model_class = {"MLP": MLP, "FourierPINN": FourierPINN, "DGMNetwork": DGMNetwork}[model_name]
    model, params = init_model(model_class, key, FrozenDict(cfg_dict))
    print(f"Model: {model_name} (width={cfg_dict['model'].get('width')}, depth={cfg_dict['model'].get('depth')})")

    key, k1, k2, k3, k4 = random.split(key, 5)

    results = {}

    # 1. Batch size benchmark
    results["batch_sizes"] = benchmark_jacobian_batch_sizes(model, params, k1, dtype)

    # 2. jacfwd vs jacrev
    results["jacfwd_vs_jacrev"] = benchmark_jacfwd_vs_jacrev(model, params, k2, dtype)

    # 3. Architecture comparison
    results["architectures"] = benchmark_architectures(cfg_dict, k3, dtype)

    # 4. Loss component breakdown
    results["loss_components"] = benchmark_loss_components(model, params, k4, cfg_dict, dtype)

    # Save results
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        report_path = os.path.join(args.output, "pde_jacobian_benchmark.json")
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nReport saved to {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
