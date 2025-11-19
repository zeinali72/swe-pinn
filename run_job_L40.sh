#!/bin/bash
set -e

# --- 1. CONFIGURATION ---
SRC_DIR="/workspace"
WORK_DIR="/tmp/hpo_fast"
S3_RESULTS="/workspace/optimisation/results"

# Setup Fast Environment (NVMe)
echo "--- 🚀 Setting up Fast Environment ---"
# Fix Matplotlib cache issue
export MPLCONFIGDIR="/tmp/matplotlib"
mkdir -p "$WORK_DIR/database" "$WORK_DIR/logs" "$MPLCONFIGDIR"
mkdir -p "$S3_RESULTS/database" "$S3_RESULTS/logs"

# Copy everything to /tmp for speed (Data is small, so this is instant)
cp -r "$SRC_DIR/src" "$WORK_DIR/"
cp -r "$SRC_DIR/optimisation" "$WORK_DIR/"
cp -r "$SRC_DIR/data" "$WORK_DIR/"

cd "$WORK_DIR"

# --- 4. RUN JOBS (Sequential) ---

# JOB 1: MLP
echo "▶️ [1/3] Starting MLP..."
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_mlp_datafree_static_BUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-mlp-building" \
  --storage "sqlite:///database/hpo-sensitivity-mlp-building.db" \
  2>&1 | tee "logs/hpo-sensitivity-mlp-building.log"

# Copy files after Job 1
cp -u database/*.db "$S3_RESULTS/database/" 2>/dev/null || true
cp -u logs/*.log "$S3_RESULTS/logs/" 2>/dev/null || true
echo "✅ Files copied to S3 results."

# JOB 2: Fourier
echo "▶️ [2/3] Starting Fourier..."
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_fourier_datafree_static_BUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-fourier-building" \
  --storage "sqlite:///database/hpo-sensitivity-fourier-building.db" \
  2>&1 | tee "logs/hpo-sensitivity-fourier-building.log"

# Copy files after Job 2
cp -u database/*.db "$S3_RESULTS/database/" 2>/dev/null || true
cp -u logs/*.log "$S3_RESULTS/logs/" 2>/dev/null || true
echo "✅ Files copied to S3 results."

# JOB 3: DGM
echo "▶️ [3/3] Starting DGM..."
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_dgm_datafree_static_NOBUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-dgm-nobuilding" \
  --storage "sqlite:///database/hpo-sensitivity-dgm-nobuilding.db" \
  2>&1 | tee "logs/hpo-sensitivity-dgm-nobuilding.log"

echo "--- 🎉 All Success ---"