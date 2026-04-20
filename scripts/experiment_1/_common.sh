#!/usr/bin/env bash
# Shared helpers for Experiment 1 ablation row scripts.
# Sourced, not executed directly.
#
# Usage inside a row script:
#   source "$(dirname "$0")/_exp1_ablation_common.sh"
#   run_ablation_row "A0" "experiments.experiment_1.train" \
#                    "configs/experiment_1/experiment_1.yaml"
#
# Env overrides:
#   SEED              (default 42)   — training.seed override
#   WANDB_RUN_GROUP   (default exp1-ablation-v1)
#   EXTRA_TAGS        (default '')   — appended to WANDB_TAGS

set -euo pipefail

run_ablation_row() {
    local row="$1"
    local module="$2"
    local base_config="$3"

    local seed="${SEED:-42}"
    local group="${WANDB_RUN_GROUP:-exp1-ablation-v1}"
    local extra="${EXTRA_TAGS:-}"

    if [[ ! -f "$base_config" ]]; then
        echo "ERROR: config not found: $base_config" >&2
        return 1
    fi

    local tmp_config
    tmp_config=$(mktemp --suffix=.yaml)
    trap "rm -f '$tmp_config'" RETURN

    python - "$base_config" "$tmp_config" "$seed" <<'PY'
import sys, yaml
src, dst, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg.setdefault("training", {})["seed"] = seed
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

    local tags="exp1-ablation,row=${row},seed=${seed}"
    if [[ -n "$extra" ]]; then
        tags="${tags},${extra}"
    fi

    export WANDB_RUN_GROUP="$group"
    export WANDB_TAGS="$tags"

    echo "=================================================="
    echo "Experiment 1 ablation: row=${row} seed=${seed}"
    echo "  Module : ${module}"
    echo "  Config : ${base_config}"
    echo "  W&B    : group=${group}  tags=${tags}"
    echo "=================================================="

    python -m "$module" --config "$tmp_config"
}
