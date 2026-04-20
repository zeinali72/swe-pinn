#!/usr/bin/env bash
# Ablation A5 — MSE + dense (100k) sampling. Decoupling run: branches from A1,
# not A4, to isolate whether L2 or sampling density carries the breakthrough.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
source "$HERE/_common.sh"

run_ablation_row \
    "A5" \
    "experiments.experiment_1.train_nondim" \
    "configs/experiment_1/experiment_1_nondim_mse_dense.yaml"
