#!/bin/bash
set -e

# --- 1. SETUP PATHS ---
LOCAL_BASE="/tmp/hpo_fast"
LOCAL_DB_DIR="$LOCAL_BASE/database"
LOCAL_LOG_DIR="$LOCAL_BASE/logs"

# Persistent S3 paths
S3_DB_DIR="/workspace/optimisation/results/database"
S3_LOG_DIR="/workspace/optimisation/results/logs"

echo "--- 🔧 Setting up directories ---"
mkdir -p "$LOCAL_DB_DIR" "$LOCAL_LOG_DIR"
mkdir -p "$S3_DB_DIR" "$S3_LOG_DIR"

# --- 2. DEFINING SYNC FUNCTION ---
# We define this function to reuse it easily
sync_data() {
    echo "🔄 Syncing /tmp data to S3..."
    # We use rsync if available, otherwise cp -u. 
    # Using cp -r to ensure we catch everything.
    cp -u "$LOCAL_DB_DIR/"*.db "$S3_DB_DIR/" 2>/dev/null || true
    cp -u "$LOCAL_LOG_DIR/"*.log "$S3_LOG_DIR/" 2>/dev/null || true
    echo "✅ Sync complete."
}

# --- 3. SAFETY TRAP ---
# IMPORTANT: If the script fails (error) or exits, run sync_data immediately.
# This ensures you always get your logs in S3, even if Python crashes.
trap sync_data EXIT

# --- 4. RESTORE EXISTING DATA ---
echo "📥 checking for existing databases in S3..."
cp "$S3_DB_DIR/"*.db "$LOCAL_DB_DIR/" 2>/dev/null || echo "No existing DBs found to resume."

# --- 5. START JOBS (Sequential) ---
# Note: We use '| tee' to show logs on screen AND write to file

# JOB 1: MLP (Building)
echo "▶️ [1/3] Starting MLP (Building)..."
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_mlp_datafree_static_BUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-mlp-building" \
  --storage "sqlite:///$LOCAL_DB_DIR/hpo-sensitivity-mlp-building.db" \
  2>&1 | tee "$LOCAL_LOG_DIR/hpo-sensitivity-mlp-building.log"

# Sync immediately after Job 1 finishes safely
sync_data

# JOB 2: Fourier (Building)
echo "▶️ [2/3] Starting Fourier (Building)..."
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_fourier_datafree_static_BUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-fourier-building" \
  --storage "sqlite:///$LOCAL_DB_DIR/hpo-sensitivity-fourier-building.db" \
  2>&1 | tee "$LOCAL_LOG_DIR/hpo-sensitivity-fourier-building.log"

# Sync immediately after Job 2 finishes safely
sync_data

# JOB 3: DGM (No Building)
echo "▶️ [3/3] Starting DGM (No Building)..."
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_dgm_datafree_static_NOBUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-dgm-nobuilding" \
  --storage "sqlite:///$LOCAL_DB_DIR/hpo-sensitivity-dgm-nobuilding.db" \
  2>&1 | tee "$LOCAL_LOG_DIR/hpo-sensitivity-dgm-nobuilding.log"

echo "--- 🏁 All Jobs Finished Successfully ---"
# The 'trap' will trigger one final sync here automatically.