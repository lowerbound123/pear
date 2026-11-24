#!/usr/bin/env bash
# A single pass over 50 prompt groups, four sampled trajectories each, on a combined node.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd -P)
: "${WORK_DIR:?set WORK_DIR to the installed PEAR work directory}"
mkdir -p "$WORK_DIR/runs"
if [[ -z ${RUN:-} ]]; then RUN=$(mktemp -d "$WORK_DIR/runs/easy-train-200.XXXXXXXX"); fi
export RUN
mkdir -p "$RUN"
export EPISODES="$RUN/easy_train_50.json"
python3 - "$EPISODES" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
suites = ["libero_easy_left", "libero_easy_right"]
# Each side sees all 20 instructions. Revisit five (translation, rotation, gripper and combined)
# on the other training start. Start 2 is reserved for evaluation.
cases = [{"suite": suites, "task_id": list(range(start, 20, 2)), "trial_id": [start]} for start in (0, 1)]
for start in (0, 1):
    cases.append({"suite": suites, "task_id": [i for i in (0, 3, 8, 13, 17) if i % 2 != start], "trial_id": [start]})
spec = {"env_class": "pear.envs.libero.easy_env.EasyMotionEnv", "data_source": "suite", "env": cases}
if path.exists():
    if json.loads(path.read_text()) != spec:
        raise SystemExit(f"Different training manifest already exists at {path}")
else:
    path.write_text(json.dumps(spec, indent=2) + "\n")
PY

export BATCH=2 N=4 STEPS=25 SMOKE=0 MAX_PROMPT=4096 MAX_RESPONSE=12288
# The uncapped sampler submits exactly one training batch at the beginning of each step.
# With no warmup or group filtering it never prefetches another batch or replaces uniform-reward groups.
exec bash "$here/run_easy.sh" train \
  data.apply_chat_template_kwargs.enable_thinking=false data.val_batch_size=1 \
  run_sizes.sessions_per_replica=null actor_rollout_ref.rollout.max_num_seqs=8 \
  actor_rollout_ref.rollout.agent.num_workers=2 trainer.v1.separate_async.num_warmup_batches=0 \
  algorithm.filter_groups.enable=false trainer.val_before_train=false trainer.test_freq=-1 \
  trainer.save_freq=1 trainer.max_actor_ckpt_to_keep=2 "trainer.rollout_data_dir=$RUN/train_dumps" "$@"
