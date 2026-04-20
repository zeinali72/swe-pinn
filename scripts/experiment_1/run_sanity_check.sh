#!/usr/bin/env bash
# Sanity check: runs A0 and A1 once (seed=42) to confirm the pipeline works
# on GPU before committing to the full seed test + ablation.
#
# Expected results (from W&B history):
#   A0 (dimensional MSE, 10k): NSE ~0.5–0.76
#   A1 (nondim MSE, 10k):      NSE ~0.76
#
# If both complete without error and produce reasonable NSE, proceed to:
#   bash scripts/experiment_1/run_A1_seedtest.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.."

echo ""
echo "=============================================="
echo "  SANITY CHECK: A0 (dimensional baseline)"
echo "=============================================="
EXTRA_TAGS="sanity-check" bash "$HERE/run_A0.sh"

echo ""
echo "=============================================="
echo "  SANITY CHECK: A1 (nondim MSE baseline)"
echo "=============================================="
EXTRA_TAGS="sanity-check" bash "$HERE/run_A1.sh"

echo ""
echo "Sanity check complete. Both runs finished without error."
echo "Next step: bash scripts/experiment_1/run_A1_seedtest.sh"
