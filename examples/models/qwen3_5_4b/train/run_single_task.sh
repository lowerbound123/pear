#!/usr/bin/env bash
# Train Qwen3.5-4B on one LIBERO task on two A800 GPUs. With RAY_ADDRESS set, use an existing combined node.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../../.." && pwd -P)
export WORK_DIR=${WORK_DIR:?set WORK_DIR to the installed PEAR work directory}
suite=${SUITE:-libero_spatial}
task=${TASK_ID:-0}
case "$suite" in
  libero_spatial | libero_object | libero_goal | libero_10) ;;
  *) echo "Choose one suite: libero_spatial, libero_object, libero_goal or libero_10" >&2; exit 2 ;;
esac
if [[ ! $task =~ ^[0-9]$ ]] || (( $# )); then
  echo "Use one TASK_ID from 0 to 9 and no positional arguments." >&2
  exit 2
fi

export STEPS=${STEPS:-20}
export SMOKE=${SMOKE:-0}
if [[ ! $STEPS =~ ^[1-9][0-9]*$ || ! $SMOKE =~ ^[01]$ ]]; then
  echo "STEPS must be a positive integer and SMOKE must be 0 or 1." >&2
  exit 2
fi
if [[ $SMOKE == 1 ]]; then export STEPS=1; fi
ray=${VERL_ENV:-$WORK_DIR/envs/verl}/bin/ray
if [[ ! -x $ray || ! -f ${MODEL:-$WORK_DIR/models/Qwen3.5-4B}/config.json ]]; then
  echo "Check WORK_DIR: the installed Ray environment and Qwen3.5-4B weights are required." >&2
  exit 1
fi

# A fresh run has no replay buffer or checkpoint from a previous task.
mkdir -p "$WORK_DIR/runs"
RUN=$(mktemp -d "$WORK_DIR/runs/${suite}-task${task}.XXXXXXXX")
export RUN
write_episodes() {
  cat > "$1" <<JSON
{
  "env_class": "pear.envs.libero.env.LiberoEnv",
  "data_source": "suite",
  "env": {"suite": ["$suite"], "task_id": [$task], "trial_id": "$2"}
}
JSON
}
export EPISODES=$RUN/train.json
write_episodes "$EPISODES" 0-39
write_episodes "$RUN/val.json" 40-49
echo "Training $suite/task_$task for $STEPS step(s). Run directory: $RUN"

if [[ -z ${RAY_ADDRESS:-} ]]; then
  port=${PORT:-6379}
  if (: < "/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    echo "Port $port is in use. To use an existing combined Ray node, set RAY_ADDRESS to its address." >&2
    exit 1
  fi
  LIBERO_SERVERS_PER_NODE=${LIBERO_SERVERS_PER_NODE:-2} \
    bash "$ROOT/scripts/start_ray_node.sh" combined > "$RUN/ray.log" 2>&1 &
  ray_pid=$!
  stop_own_ray() {
    kill -TERM "$ray_pid" 2>/dev/null || true
    wait "$ray_pid" 2>/dev/null || true
  }
  trap stop_own_ray EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  export RAY_ADDRESS=127.0.0.1:$port
  deadline=$((SECONDS + 300))
  until timeout 10 "$ray" status --address="$RAY_ADDRESS" >/dev/null 2>&1; do
    if ! kill -0 "$ray_pid" 2>/dev/null || (( SECONDS >= deadline )); then
      tail -n 30 "$RUN/ray.log" >&2
      echo "Ray failed to become ready. See $RUN/ray.log" >&2
      exit 1
    fi
    sleep 2
  done
fi

bash "$ROOT/examples/models/qwen3_5_4b/train/run_libero.sh" \
  "data.val_files=[$RUN/val.json]" data.val_batch_size=1 \
  trainer.val_before_train=false trainer.test_freq=-1 trainer.save_freq=1

checkpoint=$RUN/ckpt/global_step_$STEPS/actor
if [[ ! -d $checkpoint ]]; then
  echo "Training ended without the requested step-$STEPS checkpoint. See $RUN/train.log" >&2
  exit 1
fi
echo "Completed $suite/task_$task. Actor checkpoint: $checkpoint"
