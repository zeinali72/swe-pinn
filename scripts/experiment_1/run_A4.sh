#!/usr/bin/env bash
# Ablation A4 — A3 + dense (100k) PDE collocation sampling. Headline row.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."
source "$HERE/_common.sh"

run_ablation_row \
    "A4" \
    "experiments.experiment_1.train_nondim_l2" \
    "configs/experiment_1/experiment_1_nondim_l2_dense.yaml"
