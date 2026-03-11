import sys
import os
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from flax.core import FrozenDict
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import concurrent.futures
import subprocess
import time
import multiprocessing

# --- Configuration ---
# FIX: reliably determine project root relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

try:
    from src.config import load_config
    from src.models import init_model
    from src.data import IrregularDomainSampler
except ImportError as e:
    print(f"ModuleNotFoundError: {e}. Please check the project root and script directory.")
    sys.exit(1)

# User Settings
TRIAL_DIR_NAME = "2026-02-10_17-41_experiment_6" 
CONFIG_NAME = "experiment_6.yaml"
# Update this path if your shapefile is named differently
SHAPEFILE_PATH = os.path.join(PROJECT_ROOT, "data", "experiment_6", "2D Zones.shp")

# Animation Settings
T_START = 0.0
T_END = 21600.0   # 6 hours
DT_FRAME = 20.0   # Matches your validation_tensor.npy timestep
FRAME_DURATION = 30 # 0.03s per frame

# --- 2. Load Resources ---
config_path = os.path.join(PROJECT_ROOT, "configs", CONFIG_NAME)
cfg_dict = load_config(config_path)

# Load Domain Artifacts (for Normalization)
scenario_name = cfg_dict.get('scenario', 'experiment_6')
base_data_path = os.path.join(PROJECT_ROOT, "data", scenario_name)
artifacts_path = os.path.join(base_data_path, "domain_artifacts.npz")
if not os.path.exists(artifacts_path):
    artifacts_path = os.path.join(base_data_path, "domain.npz")

domain_sampler = IrregularDomainSampler(artifacts_path)
all_coords = domain_sampler.tri_coords.reshape(-1, 2)
x_min, y_min = np.min(all_coords, axis=0)
x_max, y_max = np.max(all_coords, axis=0)

cfg_dict['domain'] = {
    'lx': float(x_max - x_min), 'ly': float(y_max - y_min),
    'x_min': float(x_min), 'x_max': float(x_max),
    'y_min': float(y_min), 'y_max': float(y_max),
    't_final': T_END
}
cfg = FrozenDict(cfg_dict)

# --- 3. Initialize Model ---
key = jax.random.PRNGKey(0)
models_module = __import__("src.models", fromlist=[cfg["model"]["name"]])
model_class = getattr(models_module, cfg["model"]["name"])
model, _ = init_model(model_class, key, cfg)

# Load Weights
model_dir = os.path.join(PROJECT_ROOT, "models", TRIAL_DIR_NAME)
weight_files = ["model_weights.pkl", f"{TRIAL_DIR_NAME}_params.pkl", "params.pkl"]
weights_path = next((os.path.join(model_dir, f) for f in weight_files if os.path.exists(os.path.join(model_dir, f))), None)

with open(weights_path, 'rb') as f:
    params = pickle.load(f)
print("Model weights loaded.")

# --- 4. Prepare Mesh Data ---
print(f"Loading Mesh: {SHAPEFILE_PATH}")
gdf = gpd.read_file(SHAPEFILE_PATH)

# Pre-calculate centroids (Static geometry)
# You mentioned these align with the validation_tensor x,y
centroids = gdf.geometry.centroid
x_vals = jnp.array(centroids.x.values)
y_vals = jnp.array(centroids.y.values)

# JIT-compile the inference step for speed
@jax.jit
def predict_at_t(params, x, y, t):
    t_arr = jnp.full_like(x, t)
    inputs = jnp.stack([x, y, t_arr], axis=-1)
    # Output index 0 is Depth (h)
    return model.apply(params, inputs, train=False)[..., 0] 

# --- 5. Setup Animation with Improved Colors ---
fig, ax = plt.subplots(figsize=(12, 10)) # Larger figure for better detail
div = make_axes_locatable(ax)
cax = div.append_axes("right", size="5%", pad=0.1)

# Initial Plot (t=0)
print("Initializing plot...")
h_init = predict_at_t(params, x_vals, y_vals, T_START)
gdf['h'] = np.array(h_init)

# --- COLOR & PLOT SETTINGS ---
# 'Blues' colormap: White -> Light Blue -> Dark Blue
# vmin=0.0: Ensures 0 depth is white (dry)
# vmax=3.0: Set this to the expected max flood depth (e.g., 3m or 5m)
vmin, vmax = 0.0, 10.0 

plot = gdf.plot(column='h', ax=ax, cmap='Blues', vmin=vmin, vmax=vmax, 
                legend=True, cax=cax, 
                edgecolor='face', # Hides polygon edges for smoother look
                linewidth=0.1)    # Thin lines if needed

ax.set_title(f"Simulation Time: {T_START:.1f} s", fontsize=16)
ax.axis('off')
ax.set_facecolor('white') 
cax.set_ylabel("Water Depth (m)", fontsize=12)

def update(frame_idx):
    current_t = T_START + frame_idx * DT_FRAME
    if current_t > T_END:
        return
    
    # 1. Inference
    h_pred = predict_at_t(params, x_vals, y_vals, current_t)
    
    # 2. Cleanup
    # Ensure no negative depths (numerical noise) for clean plotting
    h_clean = np.maximum(np.array(h_pred), 0.0)

    # 3. Update Plot
    # GeoPandas plot returns a PatchCollection. We update its scalar array.
    plot.collections[0].set_array(h_clean)
    
    ax.set_title(f"Simulation Time: {current_t:.1f} s", fontsize=16)
    
    if frame_idx % 10 == 0:
        print(f"Rendering frame {frame_idx} (t={current_t:.1f}s)...", end='\r')
    return plot.collections

# Calculate total frames
total_frames = int((T_END - T_START) / DT_FRAME) + 1

print(f"Starting Animation: {total_frames} frames, {DT_FRAME}s step.")

ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=total_frames, 
    interval=FRAME_DURATION, # ms
    blit=False 
)

# --- 6. Save Video ---
output_file = os.path.join(PROJECT_ROOT, "results", TRIAL_DIR_NAME, "mesh_inference_blues.mp4")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

try:
    print(f"Saving to {output_file}...")
    # Use ffmpeg for MP4
    ani.save(output_file, writer='ffmpeg', fps=1000/FRAME_DURATION)
    print("Video saved successfully!")
except Exception as e:
    print(f"FFmpeg failed ({e}). Saving as GIF instead...")
    gif_file = output_file.replace(".mp4", ".gif")
    ani.save(gif_file, writer='pillow', fps=1000/FRAME_DURATION)
    print(f"GIF saved to {gif_file}")

plt.close()