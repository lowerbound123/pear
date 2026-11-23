# Qwen3.5-4B on two A800 80GB GPUs

This recipe adds LIBERO full-parameter RL training, checkpoint export and evaluation for
[`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B), the post-trained general-purpose multimodal 4B model.
It pins revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The language model is called 4B; including the vision
encoder, the checkpoint has about 4.66 billion parameters. These are the original Qwen weights, not a PEAR-trained
robot policy. The newer Qwen-Drive 4B is a driving-specific model, outside this recipe's scope.

On two A800 GPUs, standalone LIBERO evaluation has run with total context limits of 16,384 and 36,864 tokens.
The 164,096-token evaluation profile still needs a GPU test. The commands below target the Linux GPU server.

## Hardware and execution

- One Linux machine with two A800 80GB GPUs and a driver compatible with the repository's CUDA 13 environment.
- Host RAM: 256GB recommended for optimizer offload, trajectories and checkpoint export; actual peak is unmeasured.
- Training reserves one whole GPU for the FSDP2 actor and another for a standalone vLLM replica. Ray assigns their
  device IDs; TP, sequence parallelism and the actor's FSDP group size are all 1. LIBERO renders on the CPU.
- Training uses BF16, FA2 text attention and SDPA vision attention, keeping PEAR's FP32 precision handling and
  fused score-centering head. A800 uses portable kernels, with no Hopper/Blackwell backend, FA3, FA4 or megakernel.
- AdamW states stay on the host and are staged in chunks, including the first update. The staging slots must still
  accommodate the largest parameter's two moments; a chunk setting is not a strict total memory cap.
- Sampling uses native vLLM Qwen3.5, FA2, Triton GDN prefill and vLLM prefix caching; LMCache is disabled for this run.
  The shared embedding/head weights remain tied. Vision weights and `A_log` retain FP32 during sync and export.

Training starts with a 16,384-token context. Standalone evaluation uses 164,096 tokens to match the original
27B stage-one context budget. Truncation may prevent an episode from succeeding.

The 4B training and evaluation recipes allow two retries per observation if the model returns malformed action JSON
or an action that fails schema validation. The model receives a text correction in the same episode. Each retry uses
the remaining episode token budget. Set `actor_rollout_ref.rollout.custom.max_action_retries=0` when invoking
`run_libero.sh` or `eval_libero.sh` to reproduce the single-attempt protocol.

## Install on the Linux GPU server

Run from the repository root. Use the same `WORK_DIR` in each terminal:

```bash
export WORK_DIR=/srv/pear
export MODEL_FILE="$PWD/examples/models/qwen3_5_4b/qwen3_5_4b.sh"
export MODEL_SOURCE=modelscope
CUDA_ARCH=8.0 bash scripts/install.sh libero_run
```

This example downloads the [ModelScope snapshot](https://modelscope.cn/models/Qwen/Qwen3.5-4B) at
`f09dfc75214c5f2f4940ae6a26c126ce93828a9b` into `$WORK_DIR/models/Qwen3.5-4B`. Both hubs have the same weight
shards, tokenizer and processor files. Set `MODEL_SOURCE=huggingface` for the Hugging Face snapshot.
A complete recorded download from either hub is reused. The selection applies to the model download only.

The existing installer installs the pinned model, simulator, training stack and dependencies, then runs its GPU and
simulator checks. It now builds FA2 for SM80 and skips FA3/FA4 on A800. It still installs the shared stack's LMCache
package, even though this recipe does not start an LMCache sidecar. Do not invoke this installer on macOS.

See [Installation](../../../docs/getting_started/installation.md) for disk, tools and network requirements. Existing
27B installation defaults remain unchanged; selecting this `MODEL_FILE` is necessary to download the 4B weights.

## Start the single-node cluster

In terminal 1, leave Ray in the foreground:

```bash
export WORK_DIR=/srv/pear
bash scripts/start_ray_node.sh combined
```

`combined` advertises two GPU roles on the same node, with four LIBERO server slots and an 8 GiB Ray object store.
It uses the host CPU count by default; `NUM_CPUS`, `LIBERO_SERVERS_PER_NODE` and `OBJECT_STORE_GB` remain adjustable.
Launch training on this node from a second terminal. A preflight check rejects missing/busy GPUs, missing role or
simulator resources, and overrides that would request more than one GPU per pool.

## Train and resume

To train a single task on an installed two-GPU machine, run:

```bash
WORK_DIR=/srv/pear SUITE=libero_spatial TASK_ID=0 STEPS=20 \
  bash examples/models/qwen3_5_4b/train/run_single_task.sh
```

This starts a combined Ray node and releases it when the command exits. Set `RAY_ADDRESS` to reuse your own
combined node. Each launch prints its new run directory and saves the train and validation episode files there.
Both files name only the selected suite and task. Trials 0–39 supply training initial states, and 40–49 are
reserved for validation. The training command skips evaluation. Defaults are spatial task 0 and 20 optimizer
steps with the recipe's usual reward-variance filter. Set `SMOKE=1` for the one-step connectivity check.

Start with a one-step connectivity check in a new run directory:

```bash
WORK_DIR=/srv/pear RUN=/srv/pear/runs/qwen35-4b-smoke SMOKE=1 \
  bash examples/models/qwen3_5_4b/train/run_libero.sh
```

`SMOKE=1` disables reward-variance filtering so uniformly unsuccessful groups cannot block the first update. Such a
group can have zero learning signal; completing the check does not establish that the model learned a robot skill.
It uses the same GRPO and score-centering loss as normal training and still runs real simulation on the server.

Normal training uses a separate run directory:

```bash
WORK_DIR=/srv/pear RUN=/srv/pear/runs/qwen35-4b-libero \
  bash examples/models/qwen3_5_4b/train/run_libero.sh
```

Repeating that command resumes from the run's checkpoints. Keep the same one-GPU actor world size when resuming;
the existing checkpoint loader rejects incompatible world sizes. Use a new run directory when switching from a
smoke check to a fresh training run. Extra arguments are Hydra overrides, as with the other recipes.

| Setting | Default |
| --- | --- |
| `BATCH` / `N` | 2 prompt groups / 4 trajectories per group |
| `MAX_PROMPT` / `MAX_RESPONSE` | 4096 / 12288 tokens |
| `SESSIONS_PER_REPLICA` | 8 sessions, deriving 16 decoding slots and 2 agent workers |
| `STEPS` / `EPOCHS` | 1000 / 100; `SMOKE=1` sets one step |
| `EPISODES` | LIBERO goal, spatial and object training suites |
| vLLM memory utilization | 0.70 on the rollout GPU |
| TransferQueue storage units | 4 |

The actor's dynamic microbatch cap equals the context budget (16384 tokens by default). Gradient checkpointing and
activation offload are enabled. Use `MODEL` to select another local copy of the same model architecture.

Normal training keeps the reward-variance filter, with `algorithm.filter_groups.max_num_gen_batches=8`. For capped
PEAR async runs this means at most `8 * BATCH` newly judged groups per sampling wait, including the initial groups,
before reporting that too few useful groups were found. A completed batch at the boundary is accepted. Simultaneously
finishing groups may be judged together. Zero retains the original unbounded behavior. This is a group budget, not
a timeout for an individual simulator or generation request.

If the budget is exhausted, inspect rewards, parsed actions and truncation in the run logs before increasing it:

```bash
WORK_DIR=/srv/pear RUN=/srv/pear/runs/qwen35-4b-libero \
  bash examples/models/qwen3_5_4b/train/run_libero.sh algorithm.filter_groups.max_num_gen_batches=16
```

The public `actor_rollout_ref.actor.fsdp_config.use_streamed_adamw` switch accepts `true`, `false` or `null`. This
recipe sets `true`; `null` preserves the prior automatic choice for existing recipes. Explicit streaming requires
training FSDP2, manual optimizer offload, step offload and ordinary non-fused AdamW without a gradient scaler.

## Export and evaluate

After a checkpoint has been saved, export it with the existing command:

```bash
WORK_DIR=/srv/pear bash scripts/export_checkpoint.sh \
  /srv/pear/runs/qwen35-4b-libero/ckpt/global_step_1 \
  /srv/pear/models/Qwen3.5-4B-libero-step1
```

For standalone evaluation, stop training first. Stop the foreground Ray process and, on this dedicated server,
clear the old Ray node before starting it in the evaluation role:

```bash
export WORK_DIR=/srv/pear
"$WORK_DIR/envs/verl/bin/ray" stop
LIBERO_SERVERS_PER_NODE=4 OBJECT_STORE_GB=8 NUM_CPUS=$(nproc) \
  bash scripts/start_ray_node.sh rollout
```

Leave that terminal running. In another terminal, run two TP1 evaluation replicas on LIBERO's four suites:

```bash
WORK_DIR=/srv/pear RUN=/srv/pear/runs/qwen35-4b-eval-step1 \
  MODEL=/srv/pear/models/Qwen3.5-4B-libero-step1 EVAL_STEP=1 \
  bash examples/models/qwen3_5_4b/eval/eval_libero.sh
```

To evaluate the original model, set `MODEL=/srv/pear/models/Qwen3.5-4B` and `EVAL_STEP=0`. The defaults are 4,096
prompt tokens, 160,000 response tokens shared across the trajectory, thinking enabled, and four sessions per GPU.
Results are written under `RUN/val_dumps/`.

For a smaller first test, run the curriculum screening script from the pulled checkout. By default it starts its
own rollout node on the server, evaluates trial 0 for each of the 30 Goal, Spatial and Object tasks, and prints
a fresh run directory. Set `TRIALS=8` for eight initial states per task. The per-task results are in
`task_rates.tsv` inside the run directory.

```bash
export WORK_DIR=/srv/pear
bash examples/models/qwen3_5_4b/eval/screen_libero_tasks.sh
```

## Verification boundary

On macOS, only source parsing and inspection are permitted for this change: Python AST, `bash -n`, YAML parsing,
local link/path checks and `git diff --check`. Do not source the launch scripts or execute their dry-run/pytest
helpers on the Mac: those helpers run Python and subprocesses even when the GPU programs have stand-ins.

Regression test sources cover model settings, composed launches, parser behavior, role placement, shared weights,
the optimizer switch, installer routing and the filter budget. They have not been executed as part of this change.
The existing H100/GB300 expectations remain in the suite. On a suitably provisioned Linux machine, first run the
relevant CPU tests, then perform the following A800 acceptance checks:

1. The actor and sampler use different GPUs; check actual process/device placement.
2. One LIBERO group reaches sampling, forward/backward, optimizer update and NCCL weight sync without an error.
3. A saved checkpoint resumes, exports and completes a small LIBERO evaluation.
4. Measure both GPU peaks and host RAM through the first update, weight sync and checkpoint save; inspect NaN/OOM,
   reward, action-parse and truncation metrics before scaling context or concurrency.

No A800 throughput, peak-memory or learning-quality claim should be made until those checks have run.
