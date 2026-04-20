#!/bin/bash
# Regenerate validation data with hu/hv columns for experiments 2-7.
#
# The validation .npy files must contain 6 columns [t, x, y, h, u, v] so that
# the training loop can compute NSE/RMSE for hu and hv in addition to h.
#
# Prerequisites:
#   - Raw ICM simulation data must exist as val_full_domain.npy (6 columns)
#     under data/experiment_N/. If only 4-column tensors exist, re-run the
#     full pipeline from ICM CSV -> binary -> .npy first.
#   - For gauge-based experiments (3-7), gauge CSVs (gauge_depth.csv,
#     gauge_angle.csv, gauge_speed.csv) and gauge_metadata.csv must exist.
#
# This script is a convenience wrapper. See individual scripts for options.
#
# Depends on: #38 (scripts cleanup) for gauge processing pipeline.
#
# Usage:
#   ./regenerate_validation_data.sh [--val-samples N] [--train-samples N]
#                                   [--max-time T] [--seed S]
set -euo pipefail

# --- Configurable defaults (override via CLI flags) ---
VAL_SAMPLES=65536
TRAIN_SAMPLES=2000
MAX_TIME=21600
SEED=42

while [[ $# -gt 0 ]]; do
    case "$1" in
        --val-samples)  VAL_SAMPLES="$2"; shift 2 ;;
        --train-samples) TRAIN_SAMPLES="$2"; shift 2 ;;
        --max-time)     MAX_TIME="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--val-samples N] [--train-samples N] [--max-time T] [--seed S]"
            echo ""
            echo "Defaults: --val-samples $VAL_SAMPLES --train-samples $TRAIN_SAMPLES"
            echo "          --max-time $MAX_TIME --seed $SEED"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=== Regenerating validation data with hu/hv columns ==="
echo "  val_samples=$VAL_SAMPLES  train_samples=$TRAIN_SAMPLES  max_time=$MAX_TIME  seed=$SEED"
echo ""

# --- Experiment 2: Building obstacle ---
# Uses generate_training_data.py which samples from val_full_domain.npy
EXP2_TENSOR="$PROJECT_ROOT/data/experiment_2/val_full_domain.npy"
if [ -f "$EXP2_TENSOR" ]; then
    echo "[Experiment 2] Regenerating from val_full_domain.npy..."
    python "$SCRIPT_DIR/generate_training_data.py" \
        --scenario experiment_2 \
        --val_samples "$VAL_SAMPLES" \
        --train_samples "$TRAIN_SAMPLES" \
        --train_max_time "$MAX_TIME" \
        --val_max_time "$MAX_TIME" \
        --plot_time "$MAX_TIME" \
        --seed "$SEED"
    echo "[Experiment 2] Done."
else
    echo "[Experiment 2] SKIP: $EXP2_TENSOR not found."
    echo "  Run the full pipeline: ICM CSV -> binary -> .npy first."
fi

echo ""

# --- Experiments 3-7: Gauge-based validation ---
# These use process_gauge_csvs.py to convert depth/angle/speed CSVs to .npy
for exp_num in 3 4 5 6 7; do
    EXP_DIR="$PROJECT_ROOT/data/experiment_${exp_num}"
    META_FILE="$EXP_DIR/gauge_metadata.csv"

    if [ -f "$META_FILE" ]; then
        # Look for gauge CSVs: prefer canonical names, fall back to glob.
        DEPTH_FILE="$EXP_DIR/gauge_depth.csv"
        ANGLE_FILE="$EXP_DIR/gauge_angle.csv"
        SPEED_FILE="$EXP_DIR/gauge_speed.csv"
        [ -f "$DEPTH_FILE" ] || DEPTH_FILE=$(find "$EXP_DIR" -maxdepth 1 -name "*_depth.csv" -o -name "depth_*.csv" 2>/dev/null | head -1)
        [ -f "$ANGLE_FILE" ] || ANGLE_FILE=$(find "$EXP_DIR" -maxdepth 1 -name "*_angle.csv" -o -name "angle_*.csv" 2>/dev/null | head -1)
        [ -f "$SPEED_FILE" ] || SPEED_FILE=$(find "$EXP_DIR" -maxdepth 1 -name "*_speed.csv" -o -name "speed_*.csv" 2>/dev/null | head -1)

        if [ -n "$DEPTH_FILE" ] && [ -n "$ANGLE_FILE" ] && [ -n "$SPEED_FILE" ]; then
            echo "[Experiment $exp_num] Regenerating from gauge CSVs..."
            echo "  depth: $(basename "$DEPTH_FILE")"
            echo "  angle: $(basename "$ANGLE_FILE")"
            echo "  speed: $(basename "$SPEED_FILE")"
            python "$SCRIPT_DIR/process_gauge_csvs.py" \
                --meta "$META_FILE" \
                --depth "$DEPTH_FILE" \
                --angle "$ANGLE_FILE" \
                --speed "$SPEED_FILE" \
                --split \
                --output_train "$EXP_DIR/train_gauges.npy" \
                --output_val "$EXP_DIR/val_gauges_gt.npy"
            echo "[Experiment $exp_num] Done."
        else
            echo "[Experiment $exp_num] SKIP: Missing gauge CSV files in $EXP_DIR."
            echo "  Need: *_depth.csv, *_angle.csv, and *_speed.csv."
        fi
    elif [ -f "$EXP_DIR/val_full_domain.npy" ]; then
        echo "[Experiment $exp_num] Regenerating from val_full_domain.npy..."
        python "$SCRIPT_DIR/generate_training_data.py" \
            --scenario "experiment_${exp_num}" \
            --val_samples "$VAL_SAMPLES" \
            --train_samples "$TRAIN_SAMPLES" \
            --seed "$SEED"
        echo "[Experiment $exp_num] Done."
    else
        echo "[Experiment $exp_num] SKIP: No source data found in $EXP_DIR."
    fi
    echo ""
done

echo "=== Validation data regeneration complete ==="
echo ""
echo "Verify column counts with:"
echo "  python -c \"import numpy as np; d=np.load('data/experiment_N/val_lhs_points.npy'); print(d.shape)\""
echo ""
echo "Expected: (N, 6) with columns [t, x, y, h, u, v]"
