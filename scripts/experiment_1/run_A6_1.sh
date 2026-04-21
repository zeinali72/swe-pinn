#!/usr/bin/env bash
# Ablation A6.1 — MSE + dense (100k) sampling + float64. Companion to A6.
# Branches from A5 (not A6) to isolate the float64 effect under MSE.
# See docs/experiment_1/a4_dense_analysis.md §6 for rationale.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
source "$HERE/_common.sh"

run_ablation_row \
    "A6.1" \
    "experiments.experiment_1.train_nondim" \
    "configs/experiment_1/experiment_1_nondim_mse_dense_f64.yaml"
