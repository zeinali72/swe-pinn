import sys
import time
import numpy as np

print("=" * 80)
print("GPU SETUP DIAGNOSTIC")
print("=" * 80)

# 1. Check CUDA (via PyTorch)
# Note: This is checking the PyTorch install, not your main environment
print("\n1. PyTorch-CUDA Detection:")
try:
    import torch
    print(f"   ✓ PyTorch installed")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
        print(f"   ✓ CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ CUDA version: {torch.version.cuda}")
    else:
        print("   ✗ PyTorch reports CUDA is NOT available.")
except ImportError:
    print("   ✗ PyTorch not installed in this environment.")
except Exception as e:
    print(f"   ✗ PyTorch error: {e}")


# 2. Check CuPy
print("\n2. CuPy Detection:")
try:
    import cupy as cp
    print(f"   ✓ CuPy imported successfully")
    cupy_cuda_version = cp.cuda.runtime.runtimeGetVersion()
    print(f"   ✓ CuPy CUDA runtime version: {cupy_cuda_version}")
    arr = cp.array([1, 2, 3])
    print(f"   ✓ CuPy GPU array creation works: {arr}")
except ImportError as e:
    print(f"   ✗ CuPy import failed: {e}")
except Exception as e:
    print(f"   ✗ CuPy error: {e}")

# 3. Check cuDF
print("\n3. cuDF Detection:")
try:
    import cudf
    print(f"   ✓ cuDF imported successfully")
    print(f"   ✓ cuDF version: {cudf.__version__}")
    
    # Test basic operation
    df = cudf.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    print(f"   ✓ cuDF GPU DataFrame creation works")
    print(f"   ✓ DataFrame memory usage: {df.memory_usage(deep=True)}")
except ImportError as e:
    print(f"   ✗ cuDF import failed: {e}")
except Exception as e:
    print(f"   ✗ cuDF error: {e}")

# 4. Check JAX
# <<<--- NEW JAX SECTION ---
print("\n4. JAX Detection:")
try:
    import jax
    print(f"   ✓ JAX imported successfully")
    devices = jax.devices()
    print(f"   ✓ JAX devices found: {devices}")
    
    # Check if any device is a GPU
    is_gpu = any('gpu' in dev.device_kind.lower() for dev in devices)
    if is_gpu:
        print("   ✓ SUCCESS: JAX is configured to use the GPU.")
    else:
        print("   ✗ FAILURE: JAX is only seeing the CPU.")
        
except ImportError as e:
    print(f"   ✗ JAX import failed: {e}")
except Exception as e:
    print(f"   ✗ JAX error: {e}")
# <<<--- END NEW JAX SECTION ---

# 5. GPU Memory Check
print("\n5. GPU Memory Check:")
try:
    # Re-using cupy since it's already imported
    free, total = cp.cuda.Device().mem_info
    print(f"   ✓ Total GPU memory: {total / (1024**3):.2f} GB")
    print(f"   ✓ Free GPU memory: {free / (1024**3):.2f} GB")
except Exception as e:
    print(f"   ✗ GPU memory check failed (likely due to CuPy error above): {e}")

# 6. Performance test
print("\n6. GPU vs CPU Performance Test:")
n = 10_000_000

# CPU test
print(f"   Creating {n:,} element array...")
cpu_arr = np.random.rand(n).astype(np.float32)
start = time.time()
cpu_result = np.sin(cpu_arr)
cpu_time = time.time() - start
print(f"   CPU sin() time: {cpu_time*1000:.2f}ms")

# GPU test
try:
    # Re-using cupy
    gpu_arr = cp.array(cpu_arr)
    cp.cuda.Stream.null.synchronize()  # Wait for GPU
    start = time.time()
    gpu_result = cp.sin(gpu_arr)
    cp.cuda.Stream.null.synchronize()  # Wait for GPU
    gpu_time = time.time() - start
    print(f"   GPU sin() time: {gpu_time*1000:.2f}ms")
    print(f"   Speedup: {cpu_time/gpu_time:.1f}x")
except Exception as e:
    print(f"   GPU test failed (likely due to CuPy error above): {e}")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)