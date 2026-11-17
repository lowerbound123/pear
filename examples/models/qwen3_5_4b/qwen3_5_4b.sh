# shellcheck shell=bash disable=SC2034
# Sourced by the LIBERO recipes. GPU=a800, MODEL, RUN and MAX_MODEL_LEN are supplied by the recipe.
# The post-trained multimodal 4B checkpoint, pinned independently of the 27B model.
MODEL_REPO=Qwen/Qwen3.5-4B
MODEL_REVISION=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
MODEL_MODULES=pear.models.qwen3_5.tool_parser

if [ "$GPU" != a800 ]; then
  echo "Qwen3.5-4B's small-run profile requires GPU=a800 (got $GPU)" >&2
  exit 1
fi
GPUS_PER_NODE=${GPUS_PER_NODE:-2} # standalone evaluation; training sets each pool's size separately
TP=${TP:-1}
SESSIONS_PER_REPLICA=${SESSIONS_PER_REPLICA:-8}
if [ "$TP" != 1 ]; then
  echo "Qwen3.5-4B's A800 profile requires TP=1" >&2
  exit 1
fi
if ! [[ "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]] || [ "$MAX_MODEL_LEN" -gt 262144 ]; then
  echo "MAX_MODEL_LEN must be a positive integer no greater than 262144" >&2
  exit 1
fi
if ! [[ "$SESSIONS_PER_REPLICA" =~ ^[1-9][0-9]*$ ]] || [ "$SESSIONS_PER_REPLICA" -lt 4 ]; then
  echo "SESSIONS_PER_REPLICA must be at least 4 (one agent worker needs four sessions)" >&2
  exit 1
fi
if ! [[ "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPUS_PER_NODE must be a positive integer" >&2
  exit 1
fi

# 16-pixel patches, merged 2x2: the same image processor geometry as the Qwen3.5 family.
model_image_tokens() {
  local token_px=32
  local columns=$((($1 + token_px / 2) / token_px))
  local rows=$((($2 + token_px / 2) / token_px))
  echo $((columns * rows))
}

MODEL_ARGS=(
  +qwen3_5=a800
  actor_rollout_ref.model.path="$MODEL"
  actor_rollout_ref.model.trust_remote_code=True
  +data.apply_chat_template_kwargs.enable_thinking=true
)
MODEL_ROLLOUT_ARGS=(
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP"
)
MODEL_TRAINING_ARGS=(
  run_sizes.single_node_split=true
  actor_rollout_ref.model.use_liger=true
  actor_rollout_ref.model.gradient_checkpointing_retain_attention=false
  actor_rollout_ref.model.enable_activation_offload=True
  actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1
  actor_rollout_ref.actor.fsdp_config.fsdp_size=1
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$MAX_MODEL_LEN"
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$MAX_MODEL_LEN"
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload_step=True
  actor_rollout_ref.actor.fsdp_config.use_streamed_adamw=true
  +transfer_queue.backend.SimpleStorage.required_node_resource=trainer
)
MODEL_SERVER_ARGS=(
  --reasoning-parser qwen3
  --mamba-cache-mode align
)
