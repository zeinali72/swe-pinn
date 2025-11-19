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

# --- 3. BACKGROUND SYNCER (Runs in background to save progress) ---
(
    while true; do
        sleep 60
        cp "$LOCAL_DB_DIR/"*.db "$S3_DB_DIR/" && echo "☁️ Synced DBs" || echo "⚠️ DB Sync Failed"
        cp "$LOCAL_LOG_DIR/"*.log "$S3_LOG_DIR/" && echo "☁️ Synced Logs" || echo "⚠️ Log Sync Failed"
    done
) &
SYNC_PID=$!

# --- 4. START JOBS (SEQUENTIAL) ---
echo "--- 🚀 Starting 3 Sequential Optimizations ---"

# JOB 1: MLP
echo "▶️ [1/3] Starting MLP..."
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_mlp_datafree_static_BUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-mlp-building" \
  --storage "sqlite:///$LOCAL_DB_DIR/hpo-sensitivity-mlp-building.db" \
  2>&1 | tee "$LOCAL_LOG_DIR/hpo-sensitivity-mlp-building.log"
echo "✅ Finished MLP"

# JOB 2: Fourier
# This will only start AFTER Job 1 finishes
echo "▶️ [2/3] Starting Fourier..."
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_fourier_datafree_static_BUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-fourier-building" \
  --storage "sqlite:///$LOCAL_DB_DIR/hpo-sensitivity-fourier-building.db" \
  2>&1 | tee "$LOCAL_LOG_DIR/hpo-sensitivity-fourier-building.log"
echo "✅ Finished Fourier"

# JOB 3: DGM
# This will only start AFTER Job 2 finishes
echo "▶️ [3/3] Starting DGM..."
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_dgm_datafree_static_NOBUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-dgm-nobuilding" \
  --storage "sqlite:///$LOCAL_DB_DIR/hpo-sensitivity-dgm-nobuilding.db" \
  2>&1 | tee "$LOCAL_LOG_DIR/hpo-sensitivity-dgm-nobuilding.log"
echo "✅ Finished DGM"

# --- 5. CLEANUP ---
kill $SYNC_PID
echo "🏁 Final Sync to S3..."
cp "$LOCAL_DB_DIR/"*.db "$S3_DB_DIR/"
cp "$LOCAL_LOG_DIR/"*.log "$S3_LOG_DIR/"
echo "--- All Done ---"