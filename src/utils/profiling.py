"""Profiling utilities for JAX training runs."""
import time

import jax


def log_memory_usage(label=""):
    """Print current GPU memory usage for all devices."""
    for device in jax.devices():
        stats = device.memory_stats()
        if stats:
            used_mb = stats.get("bytes_in_use", 0) / 1e6
            peak_mb = stats.get("peak_bytes_in_use", 0) / 1e6
            print(f"[{label}] {device}: {used_mb:.0f} MB used, {peak_mb:.0f} MB peak")


def get_memory_stats():
    """Return memory stats dict for all devices."""
    memory_stats = {}
    for device in jax.devices():
        stats = device.memory_stats()
        if stats:
            memory_stats[str(device)] = {
                "peak_bytes": stats.get("peak_bytes_in_use", "N/A"),
                "current_bytes": stats.get("bytes_in_use", "N/A"),
            }
    return memory_stats


def check_jit_recompilation(fn, *args, **kwargs):
    """Run a function twice and warn if JIT recompiles on second call.

    Returns the result of the second call.
    """
    # First call: compiles
    t0 = time.perf_counter()
    result1 = fn(*args, **kwargs)
    jax.block_until_ready(result1)
    t1 = time.perf_counter()

    # Second call: should be cached
    result2 = fn(*args, **kwargs)
    jax.block_until_ready(result2)
    t2 = time.perf_counter()

    compile_ms = (t1 - t0) * 1000
    cached_ms = (t2 - t1) * 1000
    ratio = compile_ms / max(cached_ms, 0.001)

    print(f"First call: {compile_ms:.1f}ms | Second call: {cached_ms:.1f}ms | Ratio: {ratio:.1f}x")
    if ratio < 2.0:
        print("  WARNING: Low ratio suggests recompilation on second call!")

    return result2


class EpochTimer:
    """Context-manager-based timer for profiling epoch components."""

    def __init__(self):
        self.timings = {}

    def time(self, label):
        """Return a context manager that records wall-clock time under *label*."""
        return _TimerContext(self, label)

    def summary(self):
        """Return a dict of {label: list-of-ms} timings."""
        return dict(self.timings)

    def print_summary(self):
        """Print a formatted summary of all recorded timings."""
        print("\n=== TIMING BREAKDOWN (per epoch, ms) ===")
        for component, vals in self.timings.items():
            if vals:
                mean = sum(vals) / len(vals)
                print(
                    f"  {component:<25}: mean={mean:.1f}ms  "
                    f"min={min(vals):.1f}ms  max={max(vals):.1f}ms"
                )


class _TimerContext:
    def __init__(self, timer, label):
        self._timer = timer
        self._label = label

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed_ms = (time.perf_counter() - self._t0) * 1000
        self._timer.timings.setdefault(self._label, []).append(elapsed_ms)
        return False
