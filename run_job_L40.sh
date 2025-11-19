#!/bin/bash
set -e

# --- 1. SETUP PATHS ---
# /workspace is READ-ONLY (GitHub code)
# /workspace/optimisation/results is READ-WRITE (S3 Bucket)

# Use /tmp for fast local NVMe storage (Avoids network locking issues)
LOCAL_BASE="/tmp/hpo_fast"
LOCAL_DB_DIR="$LOCAL_BASE/database"
LOCAL_LOG_DIR="$LOCAL_BASE/logs"

# Persistent S3 paths (Must be inside the 'results' volume)
S3_DB_DIR="/workspace/optimisation/results/database"
S3_LOG_DIR="/workspace/optimisation/results/logs"

echo "--- 🔧 Setting up directories ---"
mkdir -p "$LOCAL_DB_DIR" "$LOCAL_LOG_DIR"
mkdir -p "$S3_DB_DIR" "$S3_LOG_DIR"

# --- 2. RESTORE EXISTING DATA ---
echo "🔄 Checking for existing databases in S3..."
cp "$S3_DB_DIR/"*.db "$LOCAL_DB_DIR/" 2>/dev/null || echo "No existing DBs found to resume."

# --- 3. BACKGROUND SYNCER ---
# Syncs /tmp -> S3 every 60 seconds
(
    while true; do
        sleep 60
        # Echo only if something is copied to avoid log spam
        cp -u "$LOCAL_DB_DIR/"*.db "$S3_DB_DIR/" 2>/dev/null && echo "☁️ Synced DBs to S3" || true
        cp -u "$LOCAL_LOG_DIR/"*.log "$S3_LOG_DIR/" 2>/dev/null || true
    done
) &
SYNC_PID=$!

# --- 5. START JOBS (3 Sequential) ---
echo "--- 🚀 Starting 3 Sequential Optimizations ---"

# JOB 1: MLP (Building)
echo "▶️ [1/3] Starting MLP (Building)..."
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_mlp_datafree_static_BUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-mlp-building" \
  --storage "sqlite:///$LOCAL_DB_DIR/hpo-sensitivity-mlp-building.db" \
  > "$LOCAL_LOG_DIR/hpo-sensitivity-mlp-building.log" 2>&1
echo "✅ Finished MLP (Building)"

# JOB 2: Fourier (Building)
echo "▶️ [2/3] Starting Fourier (Building)..."
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_fourier_datafree_static_BUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-fourier-building" \
  --storage "sqlite:///$LOCAL_DB_DIR/hpo-sensitivity-fourier-building.db" \
  > "$LOCAL_LOG_DIR/hpo-sensitivity-fourier-building.log" 2>&1
echo "✅ Finished Fourier (Building)"

# JOB 3: DGM (No Building)
echo "▶️ [3/3] Starting DGM (No Building)..."
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_dgm_datafree_static_NOBUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-dgm-nobuilding" \
  --storage "sqlite:///$LOCAL_DB_DIR/hpo-sensitivity-dgm-nobuilding.db" \
  > "$LOCAL_LOG_DIR/hpo-sensitivity-dgm-nobuilding.log" 2>&1
echo "✅ Finished DGM (No Building)"

# Wait for all jobs (though now sequential, this ensures any lingering processes)
wait

# --- 6. CLEANUP ---
kill $SYNC_PID
echo "🏁 Final Sync to S3..."
cp -u "$LOCAL_DB_DIR/"*.db "$S3_DB_DIR/"
cp -u "$LOCAL_LOG_DIR/"*.log "$S3_LOG_DIR/"
echo "--- All Done ---"