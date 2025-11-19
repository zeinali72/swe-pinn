#!/bin/bash
set -e

# --- 1. SETUP PATHS ---
SRC_DIR="/workspace"
LOCAL_BASE="/tmp/hpo_fast"
LOCAL_DB_DIR="$LOCAL_BASE/database"
LOCAL_LOG_DIR="$LOCAL_BASE/logs"
S3_DB_DIR="/workspace/optimisation/results/database"
S3_LOG_DIR="/workspace/optimisation/results/logs"

echo "--- 🔧 Setting up directories ---"
mkdir -p "$LOCAL_DB_DIR" "$LOCAL_LOG_DIR"
mkdir -p "$S3_DB_DIR" "$S3_LOG_DIR"

# --- 2. RESTORE EXISTING DATA ---
echo "🔄 Checking for existing databases in S3..."
cp "$S3_DB_DIR/"*.db "$LOCAL_DB_DIR/" 2>/dev/null || echo "No existing DBs found to resume."

# --- 3. BACKGROUND SYNCER ---
(
    while true; do
        sleep 60
        cp "$LOCAL_DB_DIR/"*.db "$S3_DB_DIR/" && echo "☁️ Synced DBs" || echo "⚠️ DB Sync Failed"
        cp "$LOCAL_LOG_DIR/"*.log "$S3_LOG_DIR/" && echo "☁️ Synced Logs" || echo "⚠️ Log Sync Failed"
    done
) &
SYNC_PID=$!

# --- 4. GPU CONFIGURATION (CRITICAL FOR PARALLEL) ---
# Limit each job to 45% VRAM so they fit side-by-side on the single GPU.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45

# --- 5. START JOBS (PARALLEL) ---
echo "--- 🚀 Starting MLP and Fourier in Parallel ---"

# JOB 1: MLP (Building)
(
    echo "▶️ [1/2] Starting MLP (Building)..."
    python3 -u -m optimisation.run_sensitivity_analysis \
      --config optimisation/configs/hpo_mlp_datafree_static_BUILDING.yaml \
      --n_trials 100 \
      --study_name "hpo-sensitivity-mlp-building" \
      --storage "sqlite:///$LOCAL_DB_DIR/hpo-sensitivity-mlp-building.db" \
      2>&1 | tee "$LOCAL_LOG_DIR/hpo-sensitivity-mlp-building.log"
    echo "✅ Finished MLP (Building)"
) &

# JOB 2: Fourier (Building)
(
    echo "▶️ [2/2] Starting Fourier (Building)..."
    python3 -u -m optimisation.run_sensitivity_analysis \
      --config optimisation/configs/hpo_fourier_datafree_static_BUILDING.yaml \
      --n_trials 100 \
      --study_name "hpo-sensitivity-fourier-building" \
      --storage "sqlite:///$LOCAL_DB_DIR/hpo-sensitivity-fourier-building.db" \
      2>&1 | tee "$LOCAL_LOG_DIR/hpo-sensitivity-fourier-building.log"
    echo "✅ Finished Fourier (Building)"
) &

# Wait for BOTH jobs to finish
wait

# --- 6. CLEANUP ---
kill $SYNC_PID
echo "🏁 Final Sync to S3..."
cp "$LOCAL_DB_DIR/"*.db "$S3_DB_DIR/"
cp "$LOCAL_LOG_DIR/"*.log "$S3_LOG_DIR/"
echo "--- All Done ---"