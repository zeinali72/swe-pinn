#!/usr/bin/env bash
# A1 seed test: runs the nondim MSE baseline on seeds 42, 123, 2024 sequentially.
# The canonical seed for ALL rows (A0–A6) is picked as the MEDIAN-NSE seed
# from this run (not the best — see docs/experiment_1_ablation_design.md §4.3).
#
# Why A1 (not A0)? A1 is the branching point for the entire nondim chain
# (A2–A6). A seed that's representative for A1's loss landscape is more
# informative for the downstream rows that reach 0.98 NSE.
#
# Prerequisite: run_sanity_check.sh should pass first.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."

for SEED in 42 123 2024; do
    echo ""
    echo "############################################"
    echo "# A1 seed test: seed=${SEED}"
    echo "############################################"
    SEED="$SEED" EXTRA_TAGS="seed-test" bash "$HERE/run_A1.sh"
done

echo ""
echo "A1 seed test complete. Pick the median-NSE seed on W&B and"
echo "use it via SEED=<N> bash scripts/experiment_1/run_A<row>.sh for A0–A6."
