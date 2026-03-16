#!/usr/bin/env python3
"""Benchmark host-device data transfer and validation overhead.

Usage:
    python scripts/benchmark_data_transfer.py --config configs/experiment_1.yaml
    python scripts/benchmark_data_transfer.py --config configs/experiment_1.yaml --output profiling/

Profiles:
    1. Epoch structure timing (data gen, scan, validation, logging)
    2. Data loading time (file read + host-to-device transfer)
    3. Validation inference + device-to-host transfer overhead
    4. Async overlap analysis (does data gen overlap with scan?)
"""
import argparse
import json
import os
import time
import timeit

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from flax.core import FrozenDict

from src.config import load_config, get_dtype, DTYPE
from src.models.pinn import MLP, FourierPINN, DGMNetwork
from src.models.factory import init_model
from src.metrics.accuracy import nse, rmse


def benchmark_data_loading(dtype, sizes=None):
    """Benchmark numpy -> JAX array transfer at various sizes."""
    if sizes is None:
        sizes = [1000, 5000, 10000, 42000, 100000]

    print("\n=== Data Loading: NumPy -> JAX Transfer ===")
    results = {}

    for n in sizes:
        # Simulate a validation dataset: (n, 6) for [x, y, t, h, hu, hv]
        data_np = np.random.randn(n, 6).astype(np.float32 if dtype == jnp.float32 else np.float64)

        # Benchmark numpy creation (simulates file read)
        t0 = time.perf_counter()
        _ = np.array(data_np)
        t1 = time.perf_counter()

        # Benchmark host -> device transfer
        data_jax = jnp.array(data_np, dtype=dtype)
        jax.block_until_ready(data_jax)
        t2 = time.perf_counter()

        load_ms = (t1 - t0) * 1000
        transfer_ms = (t2 - t1) * 1000
        size_mb = data_np.nbytes / 1e6

        print(f"  n={n:>6} ({size_mb:.1f} MB): copy={load_ms:.1f}ms, transfer={transfer_ms:.1f}ms")
        results[n] = {
            "size_mb": size_mb,
            "copy_ms": load_ms,
            "transfer_ms": transfer_ms,
        }

    return results


def benchmark_validation_overhead(model, params, dtype, val_sizes=None):
    """Benchmark validation inference + metric computation overhead."""
    if val_sizes is None:
        val_sizes = [1000, 5000, 10000, 42000]

    print("\n=== Validation Overhead: Inference + Metrics ===")
    results = {}

    key = random.PRNGKey(0)

    for n in val_sizes:
        val_points = random.uniform(key, (n, 3), dtype=dtype)
        h_true = random.uniform(key, (n,), dtype=dtype)

        # Warmup
        apply_jit = jax.jit(lambda pts: model.apply(params, pts, train=False))
        _ = apply_jit(val_points[:1])
        jax.block_until_ready(_)

        # Benchmark inference
        t0 = time.perf_counter()
        U_val = apply_jit(val_points)
        jax.block_until_ready(U_val)
        t1 = time.perf_counter()

        # Benchmark metrics (includes device->host via float())
        nse_val = float(nse(U_val[..., 0], h_true))
        rmse_val = float(rmse(U_val[..., 0], h_true))
        t2 = time.perf_counter()

        inference_ms = (t1 - t0) * 1000
        metrics_ms = (t2 - t1) * 1000

        print(f"  n={n:>6}: inference={inference_ms:.1f}ms, metrics+transfer={metrics_ms:.1f}ms")
        results[n] = {
            "inference_ms": inference_ms,
            "metrics_transfer_ms": metrics_ms,
        }

    return results


def benchmark_epoch_structure(model, params, dtype, cfg_dict, n_epochs=5):
    """Break down a single epoch into its timing components."""
    key = random.PRNGKey(42)

    batch_size = cfg_dict["training"].get("batch_size", 512)
    n_pde = cfg_dict.get("sampling", {}).get("n_points_pde", 1000)
    num_batches = max(1, n_pde // batch_size)

    print(f"\n=== Epoch Structure Timing ({n_epochs} epochs, {num_batches} batches) ===")

    timings = {
        "key_split": [],
        "data_generation": [],
        "forward_scan": [],
        "validation": [],
    }

    # Create mock data generation
    @jax.jit
    def mock_data_gen(epoch_key):
        return random.uniform(epoch_key, (num_batches, batch_size, 3), dtype=dtype)

    # Create mock scan body
    @jax.jit
    def mock_scan_body(carry, batch):
        p = carry
        U = model.apply(p, batch, train=False)
        loss = jnp.mean(U ** 2)
        return p, loss

    # Create mock validation
    n_val = 10000
    val_points = random.uniform(key, (n_val, 3), dtype=dtype)
    h_true = random.uniform(key, (n_val,), dtype=dtype)

    apply_jit = jax.jit(lambda pts: model.apply(params, pts, train=False))

    # Warmup
    _ = mock_data_gen(key)
    jax.block_until_ready(_)
    _ = apply_jit(val_points[:1])
    jax.block_until_ready(_)
    warmup_data = mock_data_gen(key)
    jax.block_until_ready(warmup_data)
    _, _ = jax.lax.scan(mock_scan_body, params, warmup_data)

    for epoch in range(n_epochs):
        # Key split
        t0 = time.perf_counter()
        key, epoch_key = random.split(key)
        t1 = time.perf_counter()
        timings["key_split"].append((t1 - t0) * 1000)

        # Data generation
        t0 = time.perf_counter()
        scan_inputs = mock_data_gen(epoch_key)
        jax.block_until_ready(scan_inputs)
        t1 = time.perf_counter()
        timings["data_generation"].append((t1 - t0) * 1000)

        # Scan (training batches)
        t0 = time.perf_counter()
        _, batch_losses = jax.lax.scan(mock_scan_body, params, scan_inputs)
        jax.block_until_ready(batch_losses)
        t1 = time.perf_counter()
        timings["forward_scan"].append((t1 - t0) * 1000)

        # Validation
        t0 = time.perf_counter()
        U_val = apply_jit(val_points)
        jax.block_until_ready(U_val)
        nse_val = float(nse(U_val[..., 0], h_true))
        t1 = time.perf_counter()
        timings["validation"].append((t1 - t0) * 1000)

    print(f"  {'Component':<20} {'Mean (ms)':>10} {'Min (ms)':>10} {'Max (ms)':>10}")
    print("  " + "-" * 52)
    total_mean = 0.0
    for component, vals in timings.items():
        if vals:
            mean_v = sum(vals) / len(vals)
            total_mean += mean_v
            print(f"  {component:<20} {mean_v:>10.1f} {min(vals):>10.1f} {max(vals):>10.1f}")
    print("  " + "-" * 52)
    print(f"  {'TOTAL':<20} {total_mean:>10.1f}")

    return timings


def benchmark_async_overlap(model, params, dtype, cfg_dict):
    """Test whether data generation overlaps with scan execution (async dispatch)."""
    key = random.PRNGKey(99)
    batch_size = cfg_dict["training"].get("batch_size", 512)
    n_pde = cfg_dict.get("sampling", {}).get("n_points_pde", 1000)
    num_batches = max(1, n_pde // batch_size)

    @jax.jit
    def mock_data_gen(epoch_key):
        return random.uniform(epoch_key, (num_batches, batch_size, 3), dtype=dtype)

    @jax.jit
    def mock_scan_body(carry, batch):
        p = carry
        U = model.apply(p, batch, train=False)
        loss = jnp.mean(U ** 2)
        return p, loss

    # Warmup
    d = mock_data_gen(key)
    jax.block_until_ready(d)
    _, _ = jax.lax.scan(mock_scan_body, params, d)

    print("\n=== Async Overlap Analysis ===")

    # Sequential: gen -> block -> scan -> block
    key, k1, k2 = random.split(key, 3)
    t0 = time.perf_counter()
    for i in range(5):
        ki = random.fold_in(k1, i)
        data = mock_data_gen(ki)
        jax.block_until_ready(data)
        _, losses = jax.lax.scan(mock_scan_body, params, data)
        jax.block_until_ready(losses)
    t_sequential = (time.perf_counter() - t0) * 1000

    # Pipelined: gen N+1 while scan N runs (no block_until_ready between)
    t0 = time.perf_counter()
    data = mock_data_gen(random.fold_in(k2, 0))
    for i in range(5):
        next_data = mock_data_gen(random.fold_in(k2, i + 1))
        _, losses = jax.lax.scan(mock_scan_body, params, data)
        data = next_data
    jax.block_until_ready(losses)
    t_pipelined = (time.perf_counter() - t0) * 1000

    speedup = t_sequential / max(t_pipelined, 0.001)
    print(f"  Sequential (5 epochs):  {t_sequential:.1f} ms")
    print(f"  Pipelined  (5 epochs):  {t_pipelined:.1f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    if speedup > 1.1:
        print("  -> Async dispatch provides meaningful overlap")
    else:
        print("  -> Limited overlap (sync points dominate)")

    return {
        "sequential_ms": t_sequential,
        "pipelined_ms": t_pipelined,
        "speedup": speedup,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark data transfer and validation overhead")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", default=None, help="Output directory for JSON report")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    dtype = get_dtype(cfg_dict)
    key = random.PRNGKey(cfg_dict["training"].get("seed", 42))

    print(f"Config: {args.config}")
    print(f"Devices: {jax.devices()}")
    print(f"dtype: {dtype}")

    model_name = cfg_dict["model"]["name"]
    model_class = {"MLP": MLP, "FourierPINN": FourierPINN, "DGMNetwork": DGMNetwork}[model_name]
    model, params = init_model(model_class, key, FrozenDict(cfg_dict))
    print(f"Model: {model_name}")

    results = {}

    # 1. Data loading transfer
    results["data_loading"] = benchmark_data_loading(dtype)

    # 2. Validation overhead
    results["validation_overhead"] = benchmark_validation_overhead(model, params, dtype)

    # 3. Epoch structure timing
    results["epoch_structure"] = benchmark_epoch_structure(model, params, dtype, cfg_dict)

    # 4. Async overlap analysis
    results["async_overlap"] = benchmark_async_overlap(model, params, dtype, cfg_dict)

    # Save results
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        report_path = os.path.join(args.output, "data_transfer_benchmark.json")
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nReport saved to {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
