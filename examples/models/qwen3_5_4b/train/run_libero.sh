#!/usr/bin/env bash
# One trainer GPU and one standalone rollout GPU on a combined Ray node. Extra arguments are Hydra overrides.
# SMOKE=1: a one-step connectivity check with reward-variance filtering disabled (not a learning-quality test).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd -P)

export WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
export RUN=${RUN:?the run directory}
export GPU=${GPU:-a800}
export MODEL=${MODEL:-$WORK_DIR/models/Qwen3.5-4B}
export MODEL_FILE=$ROOT/examples/models/qwen3_5_4b/qwen3_5_4b.sh
export TRAINER_NNODES=1 ROLLOUT_NNODES=1
export TRAINER_GPUS_PER_NODE=1 ROLLOUT_GPUS_PER_NODE=1
export BATCH=${BATCH:-2} N=${N:-4}
export MAX_PROMPT=${MAX_PROMPT:-4096} MAX_RESPONSE=${MAX_RESPONSE:-12288}
export SESSIONS_PER_REPLICA=${SESSIONS_PER_REPLICA:-8}
export STEPS=${STEPS:-1000}

for value in "$BATCH" "$N" "$MAX_PROMPT" "$MAX_RESPONSE" "$SESSIONS_PER_REPLICA" "$STEPS"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "batch, group, token, session and step sizes must be positive integers" >&2
    exit 1
  fi
done
if [ "$N" -lt 2 ] || [ "$SESSIONS_PER_REPLICA" -lt "$N" ]; then
  echo "GRPO needs N >= 2 and SESSIONS_PER_REPLICA >= N" >&2
  exit 1
fi

FILTER_ARGS=(algorithm.filter_groups.max_num_gen_batches=8)
case "${SMOKE:-0}" in
  0) ;;
  1)
    export STEPS=1
    FILTER_ARGS+=(algorithm.filter_groups.enable=false)
    ;;
  *) echo "SMOKE must be 0 or 1" >&2; exit 1 ;;
esac

exec bash "$ROOT/examples/models/qwen3_8_27b_pear/train/run_stage1.sh" "${FILTER_ARGS[@]}" "$@"
