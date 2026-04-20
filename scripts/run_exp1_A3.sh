#!/usr/bin/env bash
# Ablation A3 — A2 + Charbonnier epsilon stabilization (l2_eps = 1e-12).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
source "$HERE/_exp1_ablation_common.sh"

run_ablation_row \
    "A3" \
    "experiments.experiment_1.train_nondim_l2" \
    "configs/experiment_1/experiment_1_nondim_l2.yaml"
