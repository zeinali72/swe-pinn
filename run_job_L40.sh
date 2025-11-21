#!/bin/bash
set -e

# --- 1. SETUP PATHS ---
LOCAL_BASE="/tmp/hpo_fast"
LOCAL_DB_DIR="$LOCAL_BASE/database"
LOCAL_LOG_DIR="$LOCAL_BASE/logs"

# OVH/S3 Mount Paths
MOUNT_DB_DIR="/workspace/optimisation/results/database"
MOUNT_LOG_DIR="/workspace/optimisation/results/logs"

echo "--- 🔧 Setting up directories ---"
# Ensure local dirs exist; clear them if you want a guaranteed clean slate (optional)
mkdir -p "$LOCAL_DB_DIR" "$LOCAL_LOG_DIR"
mkdir -p "$MOUNT_DB_DIR" "$MOUNT_LOG_DIR"

# --- 2. BACKGROUND BACKUP (Every 10 Mins) ---
(
    while true; do
        sleep 3600
        echo "⏰ [Background] Creating 10-min snapshot on mount..."
        # Uses -u to only copy if source is newer (reduces I/O)
        cp -u "$LOCAL_DB_DIR/"*.db "$MOUNT_DB_DIR/" 2>/dev/null || true
        cp -u "$LOCAL_LOG_DIR/"*.log "$MOUNT_LOG_DIR/" 2>/dev/null || true
    done
) &
SYNC_PID=$!

# --- 3. SAFETY TRAP (Final Sync on Exit/Crash) ---
cleanup() {
    echo "--- 🏁 Job Finished or Interrupted ---"
    echo "💾 Killing background syncer and forcing FINAL save..."
    kill $SYNC_PID 2>/dev/null || true
    
    # Force full copy of all DBs and Logs to mount
    cp -f "$LOCAL_DB_DIR/"*.db "$MOUNT_DB_DIR/" 2>/dev/null || true
    cp -f "$LOCAL_LOG_DIR/"*.log "$MOUNT_LOG_DIR/" 2>/dev/null || true
    
    echo "✅ Data safely persisted to $MOUNT_DB_DIR"
}
# Run cleanup on Exit, Error, or Interrupt (Ctrl+C)
trap cleanup EXIT

# --- 4. START SEQUENTIAL JOBS ---

# === JOB 1: DGM ===
echo "--- 🚀 Starting DGM (1/2) ---"
python3 -u -m optimisation.run_sensitivity_analysis \
  --config optimisation/configs/hpo_dgm_datafree_static_NOBUILDING.yaml \
  --n_trials 100 \
  --study_name "hpo-sensitivity-dgm-nobuilding" \
  --storage "sqlite:///$LOCAL_DB_DIR/hpo-sensitivity-dgm-nobuilding.db" \
  2>&1 | tee "$LOCAL_LOG_DIR/hpo-sensitivity-dgm-nobuilding.log"


# Trap triggers automatically here upon script completion