#!/usr/bin/env bash
# Standalone evaluation: stop training and start this machine's Ray node as rollout first (two TP1 replicas).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../../.." && pwd -P)

export WORK_DIR=${WORK_DIR:?the directory scripts/install.sh built}
export RUN=${RUN:?the evaluation run directory}
export MODEL=${MODEL:?the base model or exported checkpoint directory}
export MODEL_FILE=$ROOT/examples/models/qwen3_5_4b/qwen3_5_4b.sh
export GPU=${GPU:-a800}
export GPUS_PER_NODE=${GPUS_PER_NODE:-2}
export ROLLOUT_NNODES=1
export EVAL_STEP=${EVAL_STEP:-0} # zero for the base model, checkpoint step for an exported model
export MAX_PROMPT=${MAX_PROMPT:-4096} MAX_RESPONSE=${MAX_RESPONSE:-12288}
export SESSIONS_PER_REPLICA=${SESSIONS_PER_REPLICA:-8}

exec bash "$ROOT/examples/eval/eval_libero.sh" "$@"
