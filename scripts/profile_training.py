#!/usr/bin/env python3
"""Profile a training run and output JAX/TensorBoard traces.

Usage:
    python scripts/profile_training.py --config configs/experiment_1.yaml --epochs 10 --output profile_output/

Produces:
    profile_output/
    ├── jax_trace/              # JAX XLA trace (view with TensorBoard)
    ├── timing_breakdown.json   # Per-component wall-clock times
    └── memory_snapshot.json    # Peak memory per device
"""
import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
from jax import random

from src.config import load_config, get_dtype
from src.models import create_model
from src.training import (
    create_optimizer,
    calculate_num_batches,
    extract_loss_weights,
    get_active_loss_weights,
    get_sampling_count_from_config,
)
from src.utils.profiling import EpochTimer, get_memory_stats, log_memory_usage


def profile_training(config_path, n_epochs, output_dir):
    """Run a short training session with profiling enabled."""
    os.makedirs(output_dir, exist_ok=True)

    # Load config
    cfg_dict = load_config(config_path)
    cfg = cfg_dict
    dtype = get_dtype(cfg)

    print(f"Profiling config: {config_path}")
    print(f"Epochs: {n_epochs}")
    print(f"Output: {output_dir}")
    print(f"Devices: {jax.devices()}")
    print(f"dtype: {dtype}")

    # Model setup
    model = create_model(cfg)
    key = random.PRNGKey(cfg["training"].get("seed", 42))
    key, init_key = random.split(key)

    dummy_input = jnp.ones((1, 3), dtype=dtype)
    params = model.init(init_key, dummy_input, train=False)

    optimiser = create_optimizer(cfg)
    opt_state = optimiser.init(params)

    # Sampling counts
    n_pde = get_sampling_count_from_config(cfg, "n_points_pde", default=1000)
    batch_size = cfg["training"].get("batch_size", 512)
    num_batches = calculate_num_batches(n_pde, batch_size)

    timer = EpochTimer()
    jit_first_epoch_ms = 0.0

    print(f"\n--- Starting profiling ({n_epochs} epochs) ---")

    for epoch in range(n_epochs):
        key, epoch_key = random.split(key)

        # Time data generation
        with timer.time("data_generation"):
            # Generate random training points as a simple proxy
            pde_points = random.uniform(
                epoch_key, (num_batches, batch_size, 3), dtype=dtype,
                minval=jnp.array([0.0, 0.0, 0.0]),
                maxval=jnp.array([
                    cfg["domain"]["lx"],
                    cfg["domain"]["ly"],
                    cfg["domain"]["t_final"],
                ]),
            )
            jax.block_until_ready(pde_points)

        # Time forward pass
        with timer.time("forward_pass"):
            # Use first batch for forward pass timing
            U = model.apply(params, pde_points[0], train=False)
            jax.block_until_ready(U)

        # Time gradient computation (backward pass)
        with timer.time("backward_pass"):
            def dummy_loss(p):
                out = model.apply(p, pde_points[0], train=True)
                return jnp.mean(out ** 2)

            grads = jax.grad(dummy_loss)(params)
            jax.block_until_ready(grads)

        # Time optimizer step
        with timer.time("optimizer_update"):
            updates, new_opt_state = optimiser.update(grads, opt_state, params)
            new_params = jax.tree.map(lambda p, u: p + u, params, updates)
            jax.block_until_ready(new_params)

        params = new_params
        opt_state = new_opt_state

        # Track JIT compilation time (first epoch is slow)
        if epoch == 0:
            jit_first_epoch_ms = sum(
                vals[-1] for vals in timer.timings.values() if vals
            )

        # JAX trace on epoch 2 (skip epoch 0/1 which include JIT)
        if epoch == 2:
            trace_dir = os.path.join(output_dir, "jax_trace")
            os.makedirs(trace_dir, exist_ok=True)
            try:
                jax.profiler.start_trace(trace_dir)
            except Exception as e:
                print(f"Warning: Could not start JAX trace: {e}")
        if epoch == 3:
            try:
                jax.profiler.stop_trace()
                print(f"JAX trace saved to {os.path.join(output_dir, 'jax_trace')}")
            except Exception as e:
                print(f"Warning: Could not stop JAX trace: {e}")

        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs} done")

    # Stop trace if still running (fewer than 4 epochs)
    if n_epochs <= 3:
        try:
            jax.profiler.stop_trace()
        except Exception:
            pass

    # Memory snapshot
    memory_stats = get_memory_stats()
    log_memory_usage("final")

    # Build results
    results = {
        "config": config_path,
        "n_epochs": n_epochs,
        "devices": [str(d) for d in jax.devices()],
        "jit_compilation": {
            "first_epoch_ms": jit_first_epoch_ms,
            "note": "Includes tracing + XLA compile",
        },
        "timings": timer.summary(),
        "memory": memory_stats,
    }

    # Save timing breakdown
    timing_path = os.path.join(output_dir, "timing_breakdown.json")
    with open(timing_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nTiming breakdown saved to {timing_path}")

    # Save memory snapshot
    memory_path = os.path.join(output_dir, "memory_snapshot.json")
    with open(memory_path, "w") as f:
        json.dump(memory_stats, f, indent=2, default=str)
    print(f"Memory snapshot saved to {memory_path}")

    # Print summary
    timer.print_summary()
    print(f"\nJIT first-epoch overhead: {jit_first_epoch_ms:.1f}ms")


def main():
    parser = argparse.ArgumentParser(description="Profile a PINN training run")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to profile")
    parser.add_argument("--output", default="profile_output", help="Output directory")
    args = parser.parse_args()

    profile_training(args.config, args.epochs, args.output)


if __name__ == "__main__":
    main()
