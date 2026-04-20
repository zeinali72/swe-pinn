#!/usr/bin/env bash
# Ablation A6 — A4 + float64 precision. Tests H5 (is double precision worth it?).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
source "$HERE/_exp1_ablation_common.sh"

run_ablation_row \
    "A6" \
    "experiments.experiment_1.train_nondim_l2" \
    "configs/experiment_1/experiment_1_nondim_l2_f64.yaml"
