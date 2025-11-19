#!/bin/bash
set -e

# --- 1. SETUP FAST ENVIRONMENT ---
# Define the fast local directory
WORK_DIR="/tmp/hpo_fast"
LOCAL_DB_DIR="$WORK_DIR/database"
LOCAL_LOG_DIR="$WORK_DIR/logs"

# Define persistent S3 paths
S3_DB_DIR="/workspace/optimisation/results/database"
S3_LOG_DIR="/workspace/optimisation/results/logs"

echo "--- 🚀 Initializing Environment ---"
mkdir -p "$LOCAL_DB_DIR" "$LOCAL_LOG_DIR"
mkdir -p "$S3_DB_DIR" "$S3_LOG_DIR"

# COPY CODE & DATA TO NVMe (Crucial for Speed)
# This moves the execution off S3 entirely.
echo "⚡ Copying project to local NVMe..."
cp -r /workspace/src "$WORK_DIR/"
cp -r /workspace/optimisation "$WORK_DIR/"
cp -r /workspace/data "$WORK_DIR/"

# Move into the fast directory
cd "$WORK_DIR"

# --- 2. DEFINE SAFETY SYNC ---
# This runs automatically when the script exits (Success OR Failure)
sync_results() {
    echo "💾 [$(date +'%T')] Syncing results to S3..."
    cp -u "$LOCAL_DB_DIR/"*.db "$S3_DB_DIR/" 2>/dev/null || true
    cp -u "$LOCAL_LOG_DIR/"*.log "$S3_LOG_DIR/" 2>/dev/null || true
}
# If the script crashes, this ensures you still get your logs!
trap sync_results EXIT

# --- 3. START JOBS ---

# JOB 1: MLP
echo "▶️ [1/3] Starting MLP (Building)..."
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_mlp_datafree_static_BUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-mlp-building" \
  --storage "sqlite:///$LOCAL_DB_DIR/hpo-sensitivity-mlp-building.db" \
  2>&1 | tee "$LOCAL_LOG_DIR/hpo-sensitivity-mlp-building.log"
sync_results # Intermediate sync

# JOB 2: Fourier
echo "▶️ [2/3] Starting Fourier (Building)..."
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_fourier_datafree_static_BUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-fourier-building" \
  --storage "sqlite:///$LOCAL_DB_DIR/hpo-sensitivity-fourier-building.db" \
  2>&1 | tee "$LOCAL_LOG_DIR/hpo-sensitivity-fourier-building.log"
sync_results # Intermediate sync

# JOB 3: DGM
echo "▶️ [3/3] Starting DGM (No Building)..."
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_dgm_datafree_static_NOBUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-dgm-nobuilding" \
  --storage "sqlite:///$LOCAL_DB_DIR/hpo-sensitivity-dgm-nobuilding.db" \
  2>&1 | tee "$LOCAL_LOG_DIR/hpo-sensitivity-dgm-nobuilding.log"

# Final sync happens automatically due to 'trap'
echo "--- 🏁 All Jobs Finished Successfully ---"