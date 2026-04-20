#!/usr/bin/env bash
# A0 seed test: runs the baseline on seeds 42, 123, 2024 sequentially.
# The canonical seed for A1-A6 is picked as the MEDIAN-NSE seed from this run
# (not the best — see docs/experiment_1_ablation_design.md §4.3).
#
# Each run is logged to the same W&B group (exp1-ablation-v1) with
# tag seed=<N> so the three can be compared in the UI.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."

for SEED in 42 123 2024; do
    echo ""
    echo "############################################"
    echo "# A0 seed test: seed=${SEED}"
    echo "############################################"
    SEED="$SEED" EXTRA_TAGS="seed-test" bash "$HERE/run_exp1_A0.sh"
done

echo ""
echo "A0 seed test complete. Pick the median-NSE seed on W&B and"
echo "use it via SEED=<N> bash scripts/run_exp1_A<row>.sh for A1-A6."
