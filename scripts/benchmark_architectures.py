#!/usr/bin/env python3
"""Benchmark per-step training time across MLP, FourierPINN, and DGM.

Usage:
    python scripts/benchmark_architectures.py --config configs/experiment_1.yaml
    python scripts/benchmark_architectures.py --config configs/experiment_1.yaml --output profiling/
    python scripts/benchmark_architectures.py --config configs/experiment_1.yaml --skip-scaling

Benchmarks:
    1. Per-step and per-epoch wall time for each architecture
    2. JIT compilation overhead (first epoch)
    3. GPU memory peak
    4. Scaling table: time vs width x depth for each architecture
"""
import argparse
import json
import os
import time
import statistics

import jax
import jax.numpy as jnp
import numpy as np
from jax import random, lax
from flax.core import FrozenDict
import optax

from src.config import load_config, get_dtype, DTYPE
from src.models.pinn import MLP, FourierPINN, DGMNetwork
from src.models.factory import init_model


MODEL_CLASSES = {"MLP": MLP, "FourierPINN": FourierPINN, "DGMNetwork": DGMNetwork}


def _get_peak_memory_mb():
    """Return peak memory in MB or None."""
    for device in jax.devices():
        stats = device.memory_stats()
        if stats:
            return stats.get("peak_bytes_in_use", 0) / 1e6
    return None


def _param_count(params):
    return sum(p.size for p in jax.tree.leaves(params))


def _make_config(base_cfg, arch_name, width, depth):
    """Create a config dict with the given arch/width/depth."""
    cfg = dict(base_cfg)
    model_cfg = dict(base_cfg["model"])
    model_cfg["name"] = arch_name
    model_cfg["width"] = width
    model_cfg["depth"] = depth

    if arch_name == "FourierPINN":
        model_cfg.setdefault("ff_dims", 256)
        model_cfg.setdefault("fourier_scale", 10.0)
        model_cfg.setdefault("bias_init", 0.0)

    if arch_name == "MLP":
        model_cfg.setdefault("bias_init", 0.0)

    cfg["model"] = model_cfg
    return cfg


def benchmark_architecture(cfg_dict, arch_name, width, depth, dtype, key,
                           n_warmup=10, n_measured=50, batch_size=512,
                           n_pde=10000):
    """Benchmark a single architecture configuration.

    Returns dict with per-step, per-epoch, JIT time, memory, and param count.
    """
    cfg = _make_config(cfg_dict, arch_name, width, depth)
    model_class = MODEL_CLASSES.get(arch_name)
    if model_class is None:
        return {"error": f"Unknown architecture: {arch_name}"}

    try:
        model, params = init_model(model_class, key, FrozenDict(cfg))
    except Exception as e:
        return {"error": f"Init failed: {e}"}

    n_params = _param_count(params)
    num_batches = max(1, n_pde // batch_size)

    # Setup mock training step via scan
    lr = cfg_dict["training"].get("learning_rate", 1e-3)
    optimiser = optax.adam(lr)
    opt_state = optimiser.init(params)

    @jax.jit
    def mock_data_gen(epoch_key):
        return random.uniform(epoch_key, (num_batches, batch_size, 3), dtype=dtype)

    def make_scan_body(model, optimiser):
        @jax.jit
        def scan_body(carry, batch):
            p, o_s = carry

            def loss_fn(p):
                U = model.apply(p, batch, train=True)
                # Simulate PDE loss with Jacobian
                def U_fn(pts):
                    return model.apply(p, pts, train=False)
                jac = jax.vmap(jax.jacfwd(U_fn))(batch)
                return jnp.mean(U ** 2) + jnp.mean(jac ** 2)

            loss_val, grads = jax.value_and_grad(loss_fn)(p)
            updates, new_o_s = optimiser.update(grads, o_s, p)
            new_p = optax.apply_updates(p, updates)
            return (new_p, new_o_s), loss_val

        return scan_body

    scan_body = make_scan_body(model, optimiser)

    # Warmup (includes JIT compilation)
    jit_start = time.perf_counter()
    data = mock_data_gen(key)
    jax.block_until_ready(data)
    (params, opt_state), losses = lax.scan(scan_body, (params, opt_state), data)
    jax.block_until_ready(losses)
    jit_time = time.perf_counter() - jit_start

    # Additional warmup epochs
    for i in range(n_warmup - 1):
        ki = random.fold_in(key, i + 1)
        data = mock_data_gen(ki)
        jax.block_until_ready(data)
        (params, opt_state), losses = lax.scan(scan_body, (params, opt_state), data)
        jax.block_until_ready(losses)

    # Measured epochs
    epoch_times = []
    for i in range(n_measured):
        ki = random.fold_in(key, n_warmup + i)
        t0 = time.perf_counter()
        data = mock_data_gen(ki)
        jax.block_until_ready(data)
        (params, opt_state), losses = lax.scan(scan_body, (params, opt_state), data)
        jax.block_until_ready(losses)
        epoch_times.append((time.perf_counter() - t0) * 1000)

    peak_mb = _get_peak_memory_mb()

    step_times = [et / num_batches for et in epoch_times]

    return {
        "arch": arch_name,
        "width": width,
        "depth": depth,
        "n_params": n_params,
        "num_batches": num_batches,
        "jit_compile_s": jit_time,
        "per_step_ms": {
            "mean": statistics.mean(step_times),
            "std": statistics.stdev(step_times) if len(step_times) > 1 else 0.0,
            "min": min(step_times),
            "max": max(step_times),
        },
        "per_epoch_ms": {
            "mean": statistics.mean(epoch_times),
            "std": statistics.stdev(epoch_times) if len(epoch_times) > 1 else 0.0,
            "min": min(epoch_times),
            "max": max(epoch_times),
        },
        "peak_memory_mb": peak_mb,
    }


def run_reference_benchmark(cfg_dict, dtype, key, width=256, depth=4):
    """Benchmark all three architectures at the reference configuration."""
    print(f"\n=== ARCHITECTURE BENCHMARK (width={width}, depth={depth}, batch=512) ===\n")
    print(f"{'':>15} {'Per-Step (ms)':>18} {'Per-Epoch (ms)':>18} {'JIT (s)':>10} {'Peak Mem (MB)':>15} {'Params':>10}")
    print("-" * 90)

    results = {}
    for arch_name in ["MLP", "FourierPINN", "DGMNetwork"]:
        k = random.fold_in(key, hash(arch_name) % (2**31))
        result = benchmark_architecture(
            cfg_dict, arch_name, width, depth, dtype, k,
            n_warmup=5, n_measured=20,
        )
        results[arch_name] = result

        if "error" in result:
            print(f"{arch_name:<15} {'ERROR':>18} — {result['error']}")
        else:
            step = result["per_step_ms"]
            epoch = result["per_epoch_ms"]
            print(
                f"{arch_name:<15} "
                f"{step['mean']:>7.1f} +/- {step['std']:<6.1f} "
                f"{epoch['mean']:>7.1f} +/- {epoch['std']:<6.1f} "
                f"{result['jit_compile_s']:>10.1f} "
                f"{result['peak_memory_mb'] or 'N/A':>15} "
                f"{result['n_params']:>10,}"
            )

    return results


def run_scaling_sweep(cfg_dict, dtype, key, arch_name,
                      widths=None, depths=None):
    """Benchmark scaling: per-step time vs width x depth."""
    if widths is None:
        widths = [128, 256, 512]
    if depths is None:
        depths = [3, 4, 6]

    print(f"\n=== SCALING: {arch_name} per-step (ms) ===")

    # Header
    header = f"{'':>12}" + "".join(f"{'depth=' + str(d):>12}" for d in depths)
    print(header)

    results = {}

    for width in widths:
        row = f"{'width=' + str(width):>12}"
        for depth in depths:
            k = random.fold_in(key, hash((arch_name, width, depth)) % (2**31))
            try:
                result = benchmark_architecture(
                    cfg_dict, arch_name, width, depth, dtype, k,
                    n_warmup=3, n_measured=10,
                )
                if "error" in result:
                    row += f"{'ERR':>12}"
                else:
                    step_mean = result["per_step_ms"]["mean"]
                    row += f"{step_mean:>12.1f}"
                results[f"w{width}_d{depth}"] = result
            except Exception as e:
                row += f"{'FAIL':>12}"
                results[f"w{width}_d{depth}"] = {"error": str(e)}
        print(row)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark architectures")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", default=None, help="Output directory for JSON report")
    parser.add_argument("--skip-scaling", action="store_true", help="Skip the scaling sweep")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    dtype = get_dtype(cfg_dict)
    key = random.PRNGKey(cfg_dict["training"].get("seed", 42))

    print(f"Config: {args.config}")
    print(f"Devices: {jax.devices()}")
    print(f"dtype: {dtype}")

    all_results = {}

    # 1. Reference benchmark
    key, k1 = random.split(key)
    all_results["reference"] = run_reference_benchmark(cfg_dict, dtype, k1)

    # 2. Scaling sweep
    if not args.skip_scaling:
        key, k2, k3, k4 = random.split(key, 4)
        all_results["scaling"] = {}
        for arch_name, ki in [("MLP", k2), ("FourierPINN", k3), ("DGMNetwork", k4)]:
            all_results["scaling"][arch_name] = run_scaling_sweep(
                cfg_dict, dtype, ki, arch_name
            )
    else:
        print("\n(Skipping scaling sweep)")

    # Save results
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        report_path = os.path.join(args.output, "architecture_benchmark.json")
        with open(report_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nReport saved to {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
