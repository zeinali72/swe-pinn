import sys

print("=" * 80)
print("GPU SETUP DIAGNOSTIC")
print("=" * 80)

# 1. Check CUDA
print("\n1. CUDA Detection:")
try:
    import torch
    print(f"   ✓ PyTorch installed")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
    print(f"   ✓ CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"   ✓ CUDA version: {torch.version.cuda}")
except ImportError:
    print("   ✗ PyTorch not installed")

# 2. Check CuPy
print("\n2. CuPy Detection:")
try:
    import cupy as cp
    print(f"   ✓ CuPy imported successfully")
    print(f"   ✓ CuPy CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
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
    print(f"   ✓ DataFrame:\n{df}")
except ImportError as e:
    print(f"   ✗ cuDF import failed: {e}")
except Exception as e:
    print(f"   ✗ cuDF error: {e}")

# 4. Check GPU Memory
print("\n4. GPU Memory Check:")
try:
    import cupy as cp
    free, total = cp.cuda.Device().mem_info
    print(f"   ✓ Total GPU memory: {total / (1024**3):.2f} GB")
    print(f"   ✓ Free GPU memory: {free / (1024**3):.2f} GB")
except Exception as e:
    print(f"   ✗ GPU memory check failed: {e}")

# 5. Performance test
print("\n5. GPU vs CPU Performance Test:")
import numpy as np
import time

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
    import cupy as cp
    gpu_arr = cp.array(cpu_arr)
    cp.cuda.Stream.null.synchronize()  # Wait for GPU
    start = time.time()
    gpu_result = cp.sin(gpu_arr)
    cp.cuda.Stream.null.synchronize()  # Wait for GPU
    gpu_time = time.time() - start
    print(f"   GPU sin() time: {gpu_time*1000:.2f}ms")
    print(f"   Speedup: {cpu_time/gpu_time:.1f}x")
except Exception as e:
    print(f"   GPU test failed: {e}")

print("\n" + "=" * 80)