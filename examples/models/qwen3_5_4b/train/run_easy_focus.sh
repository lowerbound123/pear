#!/usr/bin/env bash
# Continue the step-25 actor and optimizer on the five motion types that failed held-out evaluation.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd -P)
: "${WORK_DIR:?set WORK_DIR to the installed PEAR work directory}"
: "${SOURCE_CHECKPOINT:?set SOURCE_CHECKPOINT to the original global_step_25 directory}"
: "${MODEL:?set MODEL to the exported step-25 model directory}"
source_ckpt=$(realpath "$SOURCE_CHECKPOINT")
if [[ "${source_ckpt##*/}" != global_step_25 || ! -s "$source_ckpt/actor/model_world_size_1_rank_0.pt" ||
      ! -s "$source_ckpt/actor/optim_world_size_1_rank_0.pt" || ! -s "$MODEL/config.json" ]]; then
  echo "The step-25 actor, optimizer and exported model are required." >&2
  exit 1
fi
mkdir -p "$WORK_DIR/runs"
if [[ -z ${RUN:-} ]]; then RUN=$(mktemp -d "$WORK_DIR/runs/easy-focus-think.XXXXXXXX"); fi
export RUN
mkdir -p "$RUN"
export EPISODES="$RUN/focus_episodes.json"
python3 - "$EPISODES" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
cases = []
for round_id in range(10):
    for task_id in (7, 8, 9, 17, 19):
        if task_id == 19:
            suites, starts = ["libero_easy_right"], [0, 1]
        else:
            suites, starts = ["libero_easy_left", "libero_easy_right"], [round_id % 2]
        cases.append({"suite": suites, "task_id": [task_id], "trial_id": starts})
spec = {"env_class": "pear.envs.libero.easy_env.EasyMotionEnv", "data_source": "suite", "env": cases}
if path.exists():
    if json.loads(path.read_text()) != spec:
        raise SystemExit(f"Existing episode order differs: {path}")
else:
    path.write_text(json.dumps(spec, indent=2) + "\n")
PY

# First launch: restore the actor (including AdamW and scheduler) but start a fresh ordered
# dataloader and replay buffer. Subsequent launches in this RUN restore their own full checkpoint.
seed="$RUN/initial/global_step_25"
mkdir -p "$seed"
if [[ ! -e "$seed/actor" ]]; then ln -s "$source_ckpt/actor" "$seed/actor"; fi
test "$(realpath "$seed/actor")" = "$source_ckpt/actor"
resume=(trainer.resume_mode=resume_path "trainer.resume_from_path=$seed")
if [[ -s "$RUN/ckpt/latest_checkpointed_iteration.txt" ]]; then resume=(trainer.resume_mode=auto); fi

export BATCH=2 N=4 STEPS=75 EPOCHS=100 SMOKE=0 MAX_PROMPT=4096 MAX_RESPONSE=258048
export SESSIONS_PER_REPLICA=8
# One ordered batch per update; no prefetched or replacement groups can mix neighboring tasks.
exec bash "$here/run_easy.sh" train "${resume[@]}" trainer.del_local_ckpt_after_load=false \
  data.shuffle=false data.apply_chat_template_kwargs.enable_thinking=true data.val_batch_size=1 \
  run_sizes.sessions_per_replica=null actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.rollout.agent.num_workers=2 trainer.v1.separate_async.num_warmup_batches=0 \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
  algorithm.filter_groups.enable=false trainer.val_before_train=false trainer.test_freq=-1 \
  trainer.save_freq=1 trainer.max_actor_ckpt_to_keep=2 "trainer.rollout_data_dir=$RUN/train_dumps" "$@"
