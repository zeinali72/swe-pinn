#!/usr/bin/env bash
# Ablation A0 — baseline: dimensional MSE, no non-dimensionalization.
# Uses train.py (the original dimensional training script). Config matches
# §4.1 common hyperparameters so A0 → A1 isolates the non-dim effect only.
# Override seed via: SEED=123 bash scripts/experiment_1/run_A0.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
source "$HERE/_common.sh"

run_ablation_row \
    "A0" \
    "experiments.experiment_1.train" \
    "configs/experiment_1/experiment_1_A0.yaml"
