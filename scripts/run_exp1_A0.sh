#!/usr/bin/env bash
# Ablation A0 — baseline: input-normalized only, MSE, no output non-dim.
# Uses train_nondim.py with scaling.enabled=false (identity-mode fallback) so
# A0 differs from A1 by exactly one variable: non-dimensionalization.
# Override seed via: SEED=123 bash scripts/run_exp1_A0.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
source "$HERE/_exp1_ablation_common.sh"

run_ablation_row \
    "A0" \
    "experiments.experiment_1.train_nondim" \
    "configs/experiment_1/experiment_1_A0.yaml"
