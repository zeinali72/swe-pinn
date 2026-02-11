import sys
import os
import pickle
import jax
jax.config.update("jax_threading.local_devices_thread_pool.size", 1)
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')  # FIX: Force non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
from flax.core import FrozenDict
import concurrent.futures
import subprocess
import time

# --- Configuration ---
PROJECT_ROOT = os.getcwd()
sys.path.append(PROJECT_ROOT)

try:
    from src.config import load_config
    from src.models import init_model
    from src.data import IrregularDomainSampler
except ImportError:
    # Fallback if running from a different directory depth
    sys.path.append(os.path.join(PROJECT_ROOT, "zeinali72", "swe-pinn", "swe-pinn-benchmark_test_1"))
    from src.config import load_config
    from src.models import init_model
    from src.data import IrregularDomainSampler

# User inputs
TRIAL_DIR_NAME = "2026-02-10_17-41_experiment_6"
CONFIG_NAME = "experiment_6.yaml"
SHAPEFILE_PATH = os.path.join(PROJECT_ROOT, "data", "experiment_6", "2D Zones.shp")
VALIDATION_TENSOR_PATH = os.path.join(PROJECT_ROOT, "data", "experiment_6", "validation_tensor.npy")
ARTIFACTS_PATH = os.path.join(PROJECT_ROOT, "data", "experiment_6", "domain_artifacts.npz")

# Video Settings
FPS = 10
DPI = 120
T_START = 0.0
T_END = 21600.0
DT = 20.0
THRESHOLD_WET = 0.01

# Output
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", TRIAL_DIR_NAME, "frames")
VIDEO_PATH = os.path.join(PROJECT_ROOT, "results", TRIAL_DIR_NAME, "comparison_video.mp4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Global Init for Workers ---
gdf_global = None

def init_worker(shp_path):
    """Each subprocess loads its own copy of the shapefile."""
    import matplotlib
    matplotlib.use('Agg')  # FIX: Also set backend in each worker process
    global gdf_global
    gdf_global = gpd.read_file(shp_path)

# --- Rendering Function ---
def render_single_frame(args):
    """
    Renders a single frame containing 3 subplots.
    """
    idx, t, h_p, h_t = args
    global gdf_global

    filename = os.path.join(OUTPUT_DIR, f"frame_{idx:05d}.png")
    if os.path.exists(filename):
        return filename

    # FIX: Work on a copy so parallel workers don't corrupt each other's data
    gdf_local = gdf_global.copy()

    # 1. Prepare Data
    diff = np.abs(h_p - h_t)

    # Update GDF columns
    gdf_local['h_pred'] = h_p
    gdf_local['h_true'] = h_t
    gdf_local['diff'] = diff

    # 2. Setup Plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)
    fig.suptitle(f"Simulation Time: {t:.1f} s", fontsize=24, fontweight='bold')

    # Custom Background Logic
    vmin, vmax = 0.0, 3.0
    diff_max = 1.0

    def plot_mesh(ax, gdf, column, cmap, v_min, v_max, title, cbar_label):
        current_cmap = matplotlib.colormaps[cmap].with_extremes(under='lightgrey')

        kwds = {
            'cmap': current_cmap,
            'vmin': max(v_min, THRESHOLD_WET),
            'vmax': v_max,
            'legend': True,
            'legend_kwds': {'label': cbar_label, 'shrink': 0.6}
        }

        gdf.plot(column=column, ax=ax, **kwds)
        ax.set_title(title, fontsize=18)
        ax.axis('off')
        ax.set_facecolor('white')

    plot_mesh(axes[0], gdf_local, 'h_pred', 'Blues', 0.0, vmax, "Inference (Model)", "Water Depth (m)")
    plot_mesh(axes[1], gdf_local, 'h_true', 'Blues', 0.0, vmax, "Baseline (Ground Truth)", "Water Depth (m)")

    diff_cmap = matplotlib.colormaps['Reds'].with_extremes(under='white')
    kwds_diff = {
        'cmap': diff_cmap,
        'vmin': 0.05,
        'vmax': diff_max,
        'legend': True,
        'legend_kwds': {'label': "Abs Error (m)", 'shrink': 0.6}
    }
    gdf_local.plot(column='diff', ax=axes[2], **kwds_diff)
    axes[2].set_title("Difference (|Pred - True|)", fontsize=18)
    axes[2].axis('off')
    axes[2].set_facecolor('white')

    plt.savefig(filename, dpi=DPI, facecolor='white')
    plt.close(fig)
    return filename

# --- Main Execution ---
if __name__ == "__main__":
    print(f"--- Starting High-Quality Render Job ---")

    # 1. Load Config
    print("Loading Configuration...")
    config_path = os.path.join(PROJECT_ROOT, "configs", CONFIG_NAME)
    if not os.path.exists(config_path):
        config_path = os.path.join(PROJECT_ROOT, "zeinali72", "swe-pinn", "swe-pinn-benchmark_test_1", "configs", CONFIG_NAME)

    cfg_dict = load_config(config_path)

    # 1.5 Calculate Domain Extents
    print(f"Calculating Domain Extents from: {ARTIFACTS_PATH}")
    artifacts_path = ARTIFACTS_PATH
    if not os.path.exists(artifacts_path):
        artifacts_path = artifacts_path.replace("domain_artifacts.npz", "domain.npz")
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Could not find domain artifacts at {ARTIFACTS_PATH} or {artifacts_path}")

    domain_sampler = IrregularDomainSampler(artifacts_path)
    all_coords = domain_sampler.tri_coords.reshape(-1, 2)
    x_min, y_min = np.min(all_coords, axis=0)
    x_max, y_max = np.max(all_coords, axis=0)

    # FIX: Modify the dict BEFORE freezing it
    if 'domain' not in cfg_dict:
        cfg_dict['domain'] = {}
    cfg_dict['domain'].update({
        'lx': float(x_max - x_min),
        'ly': float(y_max - y_min),
        'x_min': float(x_min),
        'x_max': float(x_max),
        'y_min': float(y_min),
        'y_max': float(y_max),
        't_final': T_END
    })

    cfg = FrozenDict(cfg_dict)

    # 2. Init Model
    print("Initializing Model...")
    key = jax.random.PRNGKey(0)
    try:
        models_module = __import__("src.models", fromlist=[cfg["model"]["name"]])
        model_class = getattr(models_module, cfg["model"]["name"])
    except (ImportError, AttributeError):
        import src.models as models_module
        model_class = getattr(models_module, cfg["model"]["name"])

    # FIX: Handle different return signatures from init_model
    init_result = init_model(model_class, key, cfg)
    if isinstance(init_result, tuple):
        model = init_result[0]
        # init_result[1] might be initial params or state
    else:
        model = init_result

    # 3. WEIGHT LOADING — Fixed: use <folder_name>_params.pkl
    model_dir = os.path.join(PROJECT_ROOT, "models", TRIAL_DIR_NAME)

    # FIX: Primary weight file is folder name + _params.pkl as specified by user
    weights_filename = f"{TRIAL_DIR_NAME}_params.pkl"
    weights_path = os.path.join(model_dir, weights_filename)

    if not os.path.exists(weights_path):
        # Fallback: search for any .pkl file in the directory
        if os.path.exists(model_dir):
            available_files = os.listdir(model_dir)
            pkl_files = [f for f in available_files if f.endswith('.pkl')]
            if pkl_files:
                weights_path = os.path.join(model_dir, pkl_files[0])
                print(f"WARNING: Primary weights file '{weights_filename}' not found. "
                      f"Falling back to: {pkl_files[0]}")
            else:
                raise FileNotFoundError(
                    f"No .pkl weight files found in {model_dir}. "
                    f"Expected: {weights_filename}. Found files: {available_files}"
                )
        else:
            raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    print(f"Loading weights from: {weights_path}")
    with open(weights_path, 'rb') as f:
        params = pickle.load(f)
    print("Model & Weights Loaded Successfully.")

    # DIAGNOSTIC: Print param tree structure for debugging
    if isinstance(params, dict) or isinstance(params, FrozenDict):
        print(f"  Param keys: {list(params.keys()) if hasattr(params, 'keys') else type(params)}")
        if 'params' in (params.keys() if hasattr(params, 'keys') else []):
            # If the pickle saved {'params': ..., 'batch_stats': ...} etc.
            print("  Detected nested 'params' key — extracting inner params.")
            params = params['params']

    # 4. Load Mesh & Data
    print(f"Loading Mesh: {SHAPEFILE_PATH}")
    if not os.path.exists(SHAPEFILE_PATH):
        raise FileNotFoundError(f"Shapefile not found: {SHAPEFILE_PATH}")
    gdf = gpd.read_file(SHAPEFILE_PATH)
    centroids = gdf.geometry.centroid
    x_mesh = centroids.x.values.astype(np.float32)
    y_mesh = centroids.y.values.astype(np.float32)
    n_elements = len(gdf)
    print(f"  Mesh has {n_elements} elements.")

    print(f"Loading Validation Tensor: {VALIDATION_TENSOR_PATH}")
    if not os.path.exists(VALIDATION_TENSOR_PATH):
        raise FileNotFoundError(f"Validation tensor not found: {VALIDATION_TENSOR_PATH}")
    val_data = np.load(VALIDATION_TENSOR_PATH)
    print(f"  Validation tensor shape: {val_data.shape}")

    n_timesteps_expected = int((T_END - T_START) / DT) + 1

    if val_data.ndim == 2:
        total_rows = val_data.shape[0]
        if total_rows % n_elements == 0:
            print(f"  Reshaping flat tensor: ({total_rows}, {val_data.shape[1]}) -> ({n_timesteps_expected}, {n_elements}, {val_data.shape[1]})")
            val_data = val_data.reshape(n_timesteps_expected, n_elements, -1)
        else:
            # Truncate to the largest multiple of n_elements
            new_total_rows = (total_rows // n_elements) * n_elements
            val_data = val_data[:new_total_rows]
            n_timesteps_expected = new_total_rows // n_elements
            print(f"  Truncated validation tensor to {new_total_rows} rows (was {total_rows}) to make divisible by {n_elements}.")
            print(f"  Reshaping to: ({n_timesteps_expected}, {n_elements}, {val_data.shape[1]})")
            val_data = val_data.reshape(n_timesteps_expected, n_elements, -1)
    elif val_data.ndim == 3:
        print(f"  Validation tensor already 3D: {val_data.shape}")
        if val_data.shape[0] != n_timesteps_expected:
            print(f"  WARNING: Expected {n_timesteps_expected} timesteps but tensor has {val_data.shape[0]}. Adjusting.")
            n_timesteps_expected = val_data.shape[0]
        if val_data.shape[1] != n_elements:
            print(f"  WARNING: Expected {n_elements} elements but tensor has {val_data.shape[1]}.")

    # FIX: Print available columns to help debug indexing
    print(f"  Validation tensor final shape: {val_data.shape}")
    print(f"  Using column index 3 for water depth (h). "
          f"Available columns: 0..{val_data.shape[-1]-1}")

    if val_data.shape[-1] <= 3:
        print(f"  WARNING: Only {val_data.shape[-1]} columns available. "
              f"Using last column (index {val_data.shape[-1]-1}) for water depth.")
        H_TRUE_ALL = val_data[..., -1]
    else:
        H_TRUE_ALL = val_data[..., 3]

    # 5. Batch Inference
    print("Running Batch Inference on GPU...")
    times = np.linspace(T_START, T_END, n_timesteps_expected)
    H_PRED_ALL = []

    # FIX: Build the apply function more carefully
    # Try different calling conventions for model.apply
    x_gpu = jnp.array(x_mesh)
    y_gpu = jnp.array(y_mesh)

    # Test a single call first to determine the right calling convention
    print("  Testing model forward pass...")
    t_test = jnp.full_like(x_gpu, times[0])
    inputs_test = jnp.stack([x_gpu, y_gpu, t_test], axis=-1)

    # Try with and without train kwarg
    try:
        test_out = model.apply({'params': params}, inputs_test, train=False)
        use_train_kwarg = True
        wrap_params = True
        print(f"  Model accepts train=False, params wrapped in dict. Output shape: {test_out.shape}")
    except TypeError:
        try:
            test_out = model.apply({'params': params}, inputs_test)
            use_train_kwarg = False
            wrap_params = True
            print(f"  Model does not accept train kwarg, params wrapped. Output shape: {test_out.shape}")
        except Exception:
            try:
                test_out = model.apply(params, inputs_test, train=False)
                use_train_kwarg = True
                wrap_params = False
                print(f"  Model accepts train=False, raw params. Output shape: {test_out.shape}")
            except TypeError:
                test_out = model.apply(params, inputs_test)
                use_train_kwarg = False
                wrap_params = False
                print(f"  Model called with raw params, no train kwarg. Output shape: {test_out.shape}")

    # Determine output indexing
    if test_out.ndim == 1:
        output_idx = None  # Already scalar per element
        print("  Model output is 1D (one value per element).")
    elif test_out.ndim == 2:
        print(f"  Model output is 2D with {test_out.shape[-1]} output(s). Using index 0 for h.")
        output_idx = 0
    else:
        print(f"  Model output shape: {test_out.shape}. Using [..., 0] for h.")
        output_idx = 0

    # Build the JIT-compiled predict function based on what worked
    if wrap_params and use_train_kwarg:
        @jax.jit
        def predict_batch(x, y, t):
            inputs = jnp.stack([x, y, t], axis=-1)
            out = model.apply({'params': params}, inputs, train=False)
            return out if output_idx is None else out[..., output_idx]
    elif wrap_params and not use_train_kwarg:
        @jax.jit
        def predict_batch(x, y, t):
            inputs = jnp.stack([x, y, t], axis=-1)
            out = model.apply({'params': params}, inputs)
            return out if output_idx is None else out[..., output_idx]
    elif not wrap_params and use_train_kwarg:
        @jax.jit
        def predict_batch(x, y, t):
            inputs = jnp.stack([x, y, t], axis=-1)
            out = model.apply(params, inputs, train=False)
            return out if output_idx is None else out[..., output_idx]
    else:
        @jax.jit
        def predict_batch(x, y, t):
            inputs = jnp.stack([x, y, t], axis=-1)
            out = model.apply(params, inputs)
            return out if output_idx is None else out[..., output_idx]

    for i, t in enumerate(times):
        t_gpu = jnp.full_like(x_gpu, t)
        h_pred = predict_batch(x_gpu, y_gpu, t_gpu)
        H_PRED_ALL.append(np.array(h_pred))
        if i % 50 == 0:
            print(f"  Inference step {i}/{len(times)}...")

    H_PRED_ALL = np.array(H_PRED_ALL)
    print(f"Inference Complete. Predictions shape: {H_PRED_ALL.shape}")

    # Sanity check shapes match
    assert H_PRED_ALL.shape == H_TRUE_ALL.shape, (
        f"Shape mismatch! Predictions: {H_PRED_ALL.shape}, Ground truth: {H_TRUE_ALL.shape}"
    )

    # 6. Parallel Rendering
    # Cap workers to prevent OOM
    num_workers = min(os.cpu_count() or 4, 16)
    print(f"Starting Multi-Process Rendering on {num_workers} cores...")

    tasks = []
    for i in range(len(times)):
        tasks.append((i, times[i], H_PRED_ALL[i], H_TRUE_ALL[i]))

    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(SHAPEFILE_PATH,)
    ) as executor:
        results = list(executor.map(render_single_frame, tasks))

    print(f"\nRendering finished in {time.time() - t0:.2f}s")

    # Verify all frames were created
    missing_frames = [r for r in results if r is None or not os.path.exists(r)]
    if missing_frames:
        print(f"WARNING: {len(missing_frames)} frames failed to render!")

    # 7. Stitch Video
    print("Stitching Video with FFmpeg...")
    frame_pattern = os.path.join(OUTPUT_DIR, 'frame_%05d.png')

    # Verify at least some frames exist
    first_frame = os.path.join(OUTPUT_DIR, 'frame_00000.png')
    if not os.path.exists(first_frame):
        raise FileNotFoundError(f"No frames found at {OUTPUT_DIR}. Rendering may have failed.")

    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(FPS),
        '-i', frame_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        VIDEO_PATH
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"SUCCESS: Video saved to {VIDEO_PATH}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg Error (return code {e.returncode}):")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Install it with: sudo apt install ffmpeg")
        print(f"Frames are saved at: {OUTPUT_DIR}")