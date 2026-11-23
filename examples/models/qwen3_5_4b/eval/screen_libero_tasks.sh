#!/usr/bin/env bash
# Screen the base model on the LIBERO curriculum. Run on the installed two-A800 server.
# WORK_DIR=/srv/pear bash screen_libero_tasks.sh       # one initial state per task
# bash screen_libero_tasks.sh report <run directory>    # also works during a run
set -euo pipefail

report() {
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
env = json.loads((run / "episodes.json").read_text())["env"]
trials = set(env["trial_id"])
results = {(suite, task): {} for suite in env["suite"] for task in env["task_id"]}
dump = run / "val_dumps" / "eval_0.jsonl"
summary = dump.with_name("eval_0_summary.json")
complete = summary.is_file()
if dump.exists():
    with dump.open("rb") as records:
        for line in records:
            if not complete and not line.endswith(b"\n"):
                break  # the evaluator is still writing this trajectory
            episode = json.loads(line)["extra_fields"]["episode"]
            case = episode["env"]
            key = (case["suite"], case["task_id"])
            trial = case["trial_id"]
            if key not in results or trial not in trials or trial in results[key]:
                sys.exit(f"Unexpected or duplicate trial in {dump}: {key}, trial {trial}")
            if type(episode["success"]) is not bool:
                sys.exit(f"Invalid simulator success flag in {dump}: {key}, trial {trial}")
            results[key][trial] = episode["success"]

done = sum(len(outcomes) for outcomes in results.values())
expected = len(trials) * len(results)
if complete and (done != expected or json.loads(summary.read_text())["episodes"] != expected):
    sys.exit(f"Incomplete evaluation: expected {expected} trajectories, found {done}")
print(f"{'Complete' if complete else 'Partial'}: {done}/{expected} finished trajectories", file=sys.stderr)
print("task\tsuccesses\tfinished\tplanned\tobserved_rate")
for (suite, task), outcomes in sorted(results.items(), key=lambda item: (-sum(item[1].values()), item[0])):
    hits = sum(outcomes.values())
    rate = f"{hits / len(outcomes):.1%}" if outcomes else "-"
    print(f"{suite}/task_{task}\t{hits}\t{len(outcomes)}\t{len(trials)}\t{rate}")
PY
}

if [[ ${1:-} == report && $# == 2 ]]; then
  report "$2"
  exit
elif (( $# )); then
  echo "Usage: bash $0 [report <run directory>]" >&2
  exit 2
fi

repo=$(cd "$(dirname "$0")/../../../.." && pwd -P)
export WORK_DIR=${WORK_DIR:?set WORK_DIR to the installed PEAR work directory}
export MODEL=${MODEL:-$WORK_DIR/models/Qwen3.5-4B}
export GPU=a800 GPUS_PER_NODE=2 TP=1 N=1 EVAL_STEP=0
trials=${TRIALS:-1}
include_long=${INCLUDE_LONG:-0}
if [[ ! $trials =~ ^([1-9]|[1-3][0-9]|40)$ || ! $include_long =~ ^[01]$ ]]; then
  echo "TRIALS must be 1..40 and INCLUDE_LONG must be 0 or 1." >&2
  exit 2
fi
recipe=$repo/examples/models/qwen3_5_4b/eval/eval_libero.sh
ray=${VERL_ENV:-$WORK_DIR/envs/verl}/bin/ray
if [[ ! -r $recipe || ! -x $ray || ! -s $MODEL/config.json ]]; then
  echo "Check WORK_DIR and MODEL: the evaluation recipe, Ray and model are required." >&2
  exit 1
fi
mkdir -p "$WORK_DIR/runs"
RUN=$(mktemp -d "$WORK_DIR/runs/libero-screen.XXXXXXXX")
export RUN EPISODES=$RUN/episodes.json
python3 - "$repo/pear/envs/libero/configs/libero_gso_all.json" "$EPISODES" "$trials" "$include_long" <<'PY'
import json
import sys
from pathlib import Path

source, dest, trials, include_long = sys.argv[1:]
spec = json.loads(Path(source).read_text())
suites = spec["env"]["suite"]
if include_long == "1":
    suites.append("libero_10")
# EpisodeDataset iterates in key order. Cover every task once before visiting the next initial state.
spec["env"] = {"trial_id": list(range(int(trials))), "suite": suites, "task_id": list(range(10))}
Path(dest).write_text(json.dumps(spec, indent=2) + "\n")
print(f"Run: {Path(dest).parent}\n{10 * len(suites)} tasks x {trials} initial states, one rollout each.")
PY

if [[ -z ${RAY_ADDRESS:-} ]]; then
  port=${PORT:-6379}
  if (: < "/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    echo "Port $port is in use. Set RAY_ADDRESS to reuse an existing two-GPU rollout node." >&2
    exit 1
  fi
  LIBERO_SERVERS_PER_NODE=${LIBERO_SERVERS_PER_NODE:-4} OBJECT_STORE_GB=${OBJECT_STORE_GB:-8} \
    NUM_CPUS=${NUM_CPUS:-$(nproc)} bash "$repo/scripts/start_ray_node.sh" rollout > "$RUN/ray.log" 2>&1 &
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
      tail -30 "$RUN/ray.log" >&2
      exit 1
    fi
    sleep 2
  done
fi

# Match the training policy's full-distribution sampling. Evaluation records failures as well as successes.
bash "$recipe" data.shuffle=false data.val_batch_size=1 actor_rollout_ref.rollout.top_p=1.0
if [[ ! -s $RUN/val_dumps/eval_0_summary.json ]]; then
  echo "Evaluation ended without its completion summary. Read partial results with: bash $0 report $RUN" >&2
  exit 1
fi
report "$RUN" > "$RUN/task_rates.tsv.tmp"
mv "$RUN/task_rates.tsv.tmp" "$RUN/task_rates.tsv"
cat "$RUN/task_rates.tsv"
echo "Observed at least one success:"
awk -F '\t' 'NR > 1 && $2 > 0 {print $1, $2 "/" $3; n++} END {if (!n) print "(none observed)"}' "$RUN/task_rates.tsv"
