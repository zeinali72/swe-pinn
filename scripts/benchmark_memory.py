#!/usr/bin/env python3
"""Benchmark GPU memory usage across model scales and batch sizes.

Usage:
    python scripts/benchmark_memory.py --config configs/experiment_1.yaml
    python scripts/benchmark_memory.py --config configs/experiment_1.yaml --output profiling/

Profiles:
    1. Baseline memory after model init + optimizer
    2. Peak memory during a forward Jacobian pass
    3. Scale test: memory vs batch_size x model_width x depth
    4. Per-architecture memory comparison
"""
import argparse
import json
import os

import jax
import jax.numpy as jnp
from jax import random
from flax.core import FrozenDict

from src.config import load_config, get_dtype, DTYPE
from src.models.pinn import MLP, FourierPINN, DGMNetwork
from src.models.factory import init_model


def _get_memory_bytes():
    """Return current and peak bytes in use on the first device, or None."""
    for device in jax.devices():
        stats = device.memory_stats()
        if stats:
            return {
                "current_bytes": stats.get("bytes_in_use", 0),
                "peak_bytes": stats.get("peak_bytes_in_use", 0),
            }
    return None


def _bytes_to_mb(b):
    return b / 1e6


def _param_count(params):
    """Count total parameters."""
    return sum(p.size for p in jax.tree.leaves(params))


def benchmark_init_memory(model_class, key, cfg_dict):
    """Measure memory after model init and optimizer creation."""
    import optax

    model, params = init_model(model_class, key, FrozenDict(cfg_dict))
    n_params = _param_count(params)

    mem_after_init = _get_memory_bytes()

    lr = cfg_dict["training"].get("learning_rate", 1e-3)
    optimiser = optax.adam(lr)
    opt_state = optimiser.init(params)

    mem_after_opt = _get_memory_bytes()

    return {
        "n_params": n_params,
        "mem_after_init": mem_after_init,
        "mem_after_optimizer": mem_after_opt,
        "model": model,
        "params": params,
    }


def benchmark_jacobian_memory(model, params, key, dtype, batch_sizes=None):
    """Measure peak memory during Jacobian computation at various batch sizes."""
    if batch_sizes is None:
        batch_sizes = [256, 512, 1024, 2048, 4096]

    def U_fn(pts):
        return model.apply({'params': params['params']}, pts, train=False)

    jac_fn = jax.jit(jax.vmap(jax.jacfwd(U_fn)))

    # Warmup
    warmup = jnp.ones((1, 3), dtype=dtype)
    _ = jac_fn(warmup)
    jax.block_until_ready(_)

    results = {}
    print("\n=== Jacobian Memory vs Batch Size ===")
    for n in batch_sizes:
        try:
            batch = random.uniform(key, (n, 3), dtype=dtype)

            # Run Jacobian and check memory
            result = jac_fn(batch)
            jax.block_until_ready(result)

            mem = _get_memory_bytes()
            if mem:
                print(f"  batch_size={n:>5}: current={_bytes_to_mb(mem['current_bytes']):.1f} MB, "
                      f"peak={_bytes_to_mb(mem['peak_bytes']):.1f} MB")
                results[n] = mem
            else:
                print(f"  batch_size={n:>5}: memory stats unavailable (CPU mode)")
                results[n] = {"note": "CPU — no device memory stats"}
        except Exception as e:
            print(f"  batch_size={n:>5}: FAILED — {e}")
            results[n] = {"error": str(e)}

    return results


def benchmark_scale_grid(cfg_dict, key, dtype):
    """Run a grid of (batch_size, width, depth) and measure peak memory."""
    batch_sizes = [256, 512, 1024, 2048]
    widths = [128, 256, 512]
    depths = [3, 4, 6]

    print("\n=== Scale Grid: Memory vs (batch_size, width, depth) ===")
    print(f"  {'width':>6} {'depth':>6} {'batch':>6} {'params':>10} {'peak_MB':>10} {'status':>10}")
    print("  " + "-" * 60)

    results = []

    for width in widths:
        for depth in depths:
            cfg_copy = dict(cfg_dict)
            cfg_copy["model"] = {
                **cfg_dict["model"],
                "width": width,
                "depth": depth,
                "name": "MLP",
            }

            try:
                model, params = init_model(MLP, key, FrozenDict(cfg_copy))
                n_params = _param_count(params)

                def U_fn(pts):
                    return model.apply({'params': params['params']}, pts, train=False)

                jac_fn = jax.jit(jax.vmap(jax.jacfwd(U_fn)))

                # Warmup
                _ = jac_fn(jnp.ones((1, 3), dtype=dtype))
                jax.block_until_ready(_)

                for bs in batch_sizes:
                    try:
                        batch = random.uniform(key, (bs, 3), dtype=dtype)
                        result = jac_fn(batch)
                        jax.block_until_ready(result)

                        # Also measure grad (backward through Jacobian)
                        grad_fn = jax.jit(jax.grad(
                            lambda p: jnp.mean(jax.vmap(jax.jacfwd(
                                lambda pts: model.apply({'params': p['params']}, pts, train=False)
                            ))(batch) ** 2)
                        ))
                        _ = grad_fn(params)
                        jax.block_until_ready(_)

                        mem = _get_memory_bytes()
                        peak_mb = _bytes_to_mb(mem["peak_bytes"]) if mem else -1
                        status = "OK"

                        print(f"  {width:>6} {depth:>6} {bs:>6} {n_params:>10} {peak_mb:>10.1f} {status:>10}")
                        results.append({
                            "width": width, "depth": depth, "batch_size": bs,
                            "n_params": n_params, "peak_mb": peak_mb, "status": status,
                        })
                    except Exception as e:
                        err_msg = str(e)[:40]
                        print(f"  {width:>6} {depth:>6} {bs:>6} {n_params:>10} {'N/A':>10} {'OOM' if 'memory' in err_msg.lower() else 'ERR':>10}")
                        results.append({
                            "width": width, "depth": depth, "batch_size": bs,
                            "n_params": n_params, "peak_mb": None, "status": "OOM" if "memory" in err_msg.lower() else "error",
                            "error": err_msg,
                        })
            except Exception as e:
                print(f"  {width:>6} {depth:>6} {'—':>6} {'—':>10} {'—':>10} INIT_FAIL")

    return results


def benchmark_architectures_memory(cfg_dict, key, dtype, n_points=1000):
    """Compare memory usage across architectures."""
    base_model_cfg = {
        "width": cfg_dict["model"].get("width", 128),
        "depth": cfg_dict["model"].get("depth", 3),
        "output_dim": cfg_dict["model"].get("output_dim", 3),
        "bias_init": cfg_dict["model"].get("bias_init", 0.0),
    }

    arch_configs = {
        "MLP": (MLP, {**base_model_cfg, "name": "MLP"}),
        "FourierPINN": (FourierPINN, {
            **base_model_cfg, "name": "FourierPINN",
            "ff_dims": cfg_dict["model"].get("ff_dims", 64),
            "fourier_scale": cfg_dict["model"].get("fourier_scale", 1.0),
        }),
        "DGM": (DGMNetwork, {**base_model_cfg, "name": "DGMNetwork"}),
    }

    batch = random.uniform(key, (n_points, 3), dtype=dtype)

    print(f"\n=== Per-Architecture Memory (n={n_points}) ===")
    results = {}

    for arch_name, (model_class, model_cfg) in arch_configs.items():
        try:
            cfg_copy = dict(cfg_dict)
            cfg_copy["model"] = model_cfg
            model, params = init_model(model_class, key, FrozenDict(cfg_copy))
            n_params = _param_count(params)

            def U_fn(pts):
                return model.apply({'params': params['params']}, pts, train=False)

            jac_fn = jax.jit(jax.vmap(jax.jacfwd(U_fn)))
            _ = jac_fn(batch[:1])
            jax.block_until_ready(_)

            _ = jac_fn(batch)
            jax.block_until_ready(_)

            mem = _get_memory_bytes()
            peak_mb = _bytes_to_mb(mem["peak_bytes"]) if mem else -1

            print(f"  {arch_name:<15}: {n_params:>8} params, peak={peak_mb:.1f} MB")
            results[arch_name] = {"n_params": n_params, "peak_mb": peak_mb}
        except Exception as e:
            print(f"  {arch_name:<15}: FAILED — {e}")
            results[arch_name] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU memory usage")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", default=None, help="Output directory for JSON report")
    parser.add_argument("--skip-grid", action="store_true", help="Skip the scale grid test (slow)")
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    dtype = get_dtype(cfg_dict)
    key = random.PRNGKey(cfg_dict["training"].get("seed", 42))

    print(f"Config: {args.config}")
    print(f"Devices: {jax.devices()}")
    print(f"dtype: {dtype}")

    model_name = cfg_dict["model"]["name"]
    model_class = {"MLP": MLP, "FourierPINN": FourierPINN, "DGMNetwork": DGMNetwork}[model_name]

    key, k1, k2, k3, k4 = random.split(key, 5)

    results = {}

    # 1. Init memory
    print("\n=== Baseline Memory After Init ===")
    init_result = benchmark_init_memory(model_class, k1, cfg_dict)
    n_params = init_result["n_params"]
    mem_init = init_result["mem_after_init"]
    mem_opt = init_result["mem_after_optimizer"]
    if mem_init:
        print(f"  Parameters: {n_params:,} ({n_params * 4 / 1e6:.1f} MB at float32)")
        print(f"  After model init: {_bytes_to_mb(mem_init['current_bytes']):.1f} MB")
        print(f"  After optimizer:  {_bytes_to_mb(mem_opt['current_bytes']):.1f} MB")
    else:
        print(f"  Parameters: {n_params:,} ({n_params * 4 / 1e6:.1f} MB at float32)")
        print("  Memory stats unavailable (CPU mode)")
    results["init"] = {
        "n_params": n_params,
        "mem_after_init": mem_init,
        "mem_after_optimizer": mem_opt,
    }

    # 2. Jacobian memory vs batch size
    results["jacobian_memory"] = benchmark_jacobian_memory(
        init_result["model"], init_result["params"], k2, dtype
    )

    # 3. Architecture comparison
    results["architectures"] = benchmark_architectures_memory(cfg_dict, k3, dtype)

    # 4. Scale grid
    if not args.skip_grid:
        results["scale_grid"] = benchmark_scale_grid(cfg_dict, k4, dtype)
    else:
        print("\n(Skipping scale grid test)")

    # Save results
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        report_path = os.path.join(args.output, "memory_benchmark.json")
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nReport saved to {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
