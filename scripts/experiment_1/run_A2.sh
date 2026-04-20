#!/usr/bin/env bash
# Ablation A2 — A1 + L2 loss without the Charbonnier epsilon (l2_eps = 0).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
source "$HERE/_common.sh"

run_ablation_row \
    "A2" \
    "experiments.experiment_1.train_nondim_l2" \
    "configs/experiment_1/experiment_1_nondim_l2_noeps.yaml"
