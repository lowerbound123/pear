#!/usr/bin/env bash
# Run on a rollout (eval) or combined (train) Ray node with WORK_DIR set.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
cfg=$(cd "$here/../../../pear/envs/libero/configs" && pwd -P)
: "${WORK_DIR:?set WORK_DIR to the installed work directory}"
export MODEL=${MODEL:-$WORK_DIR/models/Qwen3.5-4B} TOOL_CONFIG=$cfg/easy_tool_config.yaml
case "${1:-}" in
  eval) recipe=$here/eval/eval_libero.sh; export EPISODES=${EPISODES:-$cfg/easy_eval.json} N=${N:-1} ;;
  train) recipe=$here/train/run_libero.sh; export EPISODES=${EPISODES:-$cfg/easy_train.json} VAL_EPISODES=$cfg/easy_eval.json ;;
  *) echo "Usage: bash $0 {eval|train} [Hydra overrides]" >&2; exit 2 ;;
esac
if [[ -z ${RUN:-} ]]; then
  mkdir -p "$WORK_DIR/runs"
  RUN=$(mktemp -d "$WORK_DIR/runs/libero-easy-$1.XXXXXXXX")
  export RUN
fi
echo "Run: $RUN"
shift
exec bash "$recipe" "$@"
