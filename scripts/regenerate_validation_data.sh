#!/bin/bash
# Regenerate validation data with hu/hv columns for experiments 2-7.
#
# The validation .npy files must contain 6 columns [t, x, y, h, u, v] so that
# the training loop can compute NSE/RMSE for hu and hv in addition to h.
#
# Prerequisites:
#   - Raw ICM simulation data must exist as validation_tensor.npy (6 columns)
#     under data/experiment_N/. If only 4-column tensors exist, re-run the
#     full pipeline from ICM CSV -> binary -> .npy first.
#   - For gauge-based experiments (3-7), the gauge CSVs (depth, angle, speed)
#     and metadata CSV must be available.
#
# This script is a convenience wrapper. See individual scripts for options.
#
# Depends on: #38 (scripts cleanup) for gauge processing pipeline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Regenerating validation data with hu/hv columns ==="
echo ""

# --- Experiment 2: Building obstacle ---
# Uses generate_training_data.py which samples from validation_tensor.npy
EXP2_TENSOR="$PROJECT_ROOT/data/experiment_2/validation_tensor.npy"
if [ -f "$EXP2_TENSOR" ]; then
    echo "[Experiment 2] Regenerating from validation_tensor.npy..."
    python "$SCRIPT_DIR/generate_training_data.py" \
        --scenario experiment_2 \
        --val_samples 65536 \
        --train_samples 2000 \
        --train_max_time 21600 \
        --val_max_time 21600 \
        --plot_time 21600 \
        --seed 42
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
        # Check for gauge CSV files (naming may vary)
        DEPTH_FILE=$(find "$EXP_DIR" -name "*depth*" -name "*.csv" 2>/dev/null | head -1)
        ANGLE_FILE=$(find "$EXP_DIR" -name "*angle*" -name "*.csv" 2>/dev/null | head -1)
        SPEED_FILE=$(find "$EXP_DIR" -name "*speed*" -name "*.csv" 2>/dev/null | head -1)

        if [ -n "$DEPTH_FILE" ] && [ -n "$ANGLE_FILE" ] && [ -n "$SPEED_FILE" ]; then
            echo "[Experiment $exp_num] Regenerating from gauge CSVs..."
            python "$SCRIPT_DIR/process_gauge_csvs.py" \
                --meta "$META_FILE" \
                --depth "$DEPTH_FILE" \
                --angle "$ANGLE_FILE" \
                --speed "$SPEED_FILE" \
                --split \
                --output_train "$EXP_DIR/training_gauges.npy" \
                --output_val "$EXP_DIR/validation_gauges.npy"
            echo "[Experiment $exp_num] Done."
        else
            echo "[Experiment $exp_num] SKIP: Missing gauge CSV files in $EXP_DIR."
            echo "  Need: depth, angle, and speed CSVs."
        fi
    elif [ -f "$EXP_DIR/validation_tensor.npy" ]; then
        echo "[Experiment $exp_num] Regenerating from validation_tensor.npy..."
        python "$SCRIPT_DIR/generate_training_data.py" \
            --scenario "experiment_${exp_num}" \
            --val_samples 65536 \
            --train_samples 2000 \
            --seed 42
        echo "[Experiment $exp_num] Done."
    else
        echo "[Experiment $exp_num] SKIP: No source data found in $EXP_DIR."
    fi
    echo ""
done

echo "=== Validation data regeneration complete ==="
echo ""
echo "Verify column counts with:"
echo "  python -c \"import numpy as np; d=np.load('data/experiment_N/validation_sample.npy'); print(d.shape)\""
echo ""
echo "Expected: (N, 6) with columns [t, x, y, h, u, v]"
