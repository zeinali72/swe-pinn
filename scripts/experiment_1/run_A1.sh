#!/usr/bin/env bash
# Ablation A1 — A0 + non-dimensionalization (MSE loss, 10k sampling).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
source "$HERE/_common.sh"

run_ablation_row \
    "A1" \
    "experiments.experiment_1.train_nondim" \
    "configs/experiment_1/experiment_1_nondim.yaml"
