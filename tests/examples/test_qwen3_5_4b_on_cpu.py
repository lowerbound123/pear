import shlex
import subprocess
import sys

import pytest

from recipe_dry_run import RecipeDryRun
from repository import REPO

MODEL_FILE = REPO / "examples/models/qwen3_5_4b/qwen3_5_4b.sh"
TRAIN = "examples/models/qwen3_5_4b/train/run_libero.sh"
EVAL = "examples/models/qwen3_5_4b/eval/eval_libero.sh"


def model_settings(**variables):
    settings = {"GPU": "a800", "MODEL": "/model", "RUN": "/run", "MAX_MODEL_LEN": "16384", **variables}
    assignments = "\n".join(f"{key}={shlex.quote(value)}" for key, value in settings.items())
    command = (
        f"set -euo pipefail\n{assignments}\nsource {shlex.quote(str(MODEL_FILE))}\n"
        'printf "%s\\n" "${MODEL_ARGS[@]}" "${MODEL_ROLLOUT_ARGS[@]}" "${MODEL_TRAINING_ARGS[@]}"\n'
    )
    return subprocess.run(["bash", "-c", command], capture_output=True, text=True)


def test_model_arguments_keep_the_pinned_model_and_the_single_card_pool_sizes():
    result = model_settings()
    assert result.returncode == 0, result.stderr
    arguments = result.stdout.splitlines()
    keys = [argument.partition("=")[0].lstrip("+") for argument in arguments]
    assert len(keys) == len(set(keys))
    assert "+qwen3_5=a800" in arguments
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=1" in arguments
    assert "actor_rollout_ref.actor.fsdp_config.fsdp_size=1" in arguments
    assert "actor_rollout_ref.actor.fsdp_config.use_streamed_adamw=true" in arguments
    assert "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384" in arguments


@pytest.mark.parametrize(
    "variables", [{"TP": "2"}, {"GPU": "h100"}, {"MAX_MODEL_LEN": "262145"}, {"SESSIONS_PER_REPLICA": "1"}]
)
def test_invalid_small_run_model_settings_fail_before_launch(variables):
    assert model_settings(**variables).returncode != 0


@pytest.mark.skipif(sys.platform != "linux", reason="the shared recipes use GNU readlink")
@pytest.mark.parametrize("recipe, smoke", [(TRAIN, "0"), (TRAIN, "1"), (EVAL, "0")])
def test_a800_launches_compose_with_small_sizes_and_portable_backends(tmp_path, recipe, smoke):
    launch = RecipeDryRun(REPO, tmp_path).run(recipe, "a800", {"SMOKE": smoke})
    assert len(launch.override_keys()) == len(set(launch.override_keys()))
    config = launch.composed_config
    policy = config["actor_rollout_ref"]
    rollout = policy["rollout"]
    assert config["data"]["max_prompt_length"] == 4096
    assert rollout["kernels"] is None
    assert rollout["tensor_model_parallel_size"] == 1
    assert rollout["standalone_gpu_memory_utilization"] == 0.7
    assert rollout["fp32_modules"] == ["visual", "A_log"]
    assert rollout["custom"]["lmcache"]["enable"] is False
    assert rollout["engine_kwargs"]["vllm"]["attention_config"] == {"backend": "FLASH_ATTN", "flash_attn_version": 2}
    assert rollout["engine_kwargs"]["vllm"]["additional_config"]["gdn_prefill_backend"] == "triton"
    assert policy["actor"]["fsdp_config"]["use_gpu_backend"] is False
    assert config["transfer_queue"]["backend"]["SimpleStorage"]["num_data_storage_units"] == 4
    if recipe == TRAIN:
        assert config["data"]["max_response_length"] == 12288
        assert rollout["max_model_len"] == 16384
        assert rollout["max_num_seqs"] == 16
        assert (config["trainer"]["nnodes"], config["trainer"]["n_gpus_per_node"]) == (1, 1)
        assert (rollout["nnodes"], rollout["n_gpus_per_node"]) == (1, 1)
        assert rollout["agent"]["num_workers"] == 2
        assert config["trainer"]["v1"]["sampler"]["max_inflight_sessions"] == 8
        assert config["algorithm"]["filter_groups"]["max_num_gen_batches"] == 8
        assert config["run_sizes"]["single_node_split"] is True
        assert policy["actor"]["fsdp_config"]["use_streamed_adamw"] is True
        if smoke == "1":
            assert config["trainer"]["total_training_steps"] == 1
            assert config["algorithm"]["filter_groups"]["enable"] is False
    else:
        assert config["data"]["max_response_length"] == 160000
        assert rollout["max_model_len"] == 164096
        assert config["data"]["apply_chat_template_kwargs"]["enable_thinking"] is True
        assert rollout["n_gpus_per_node"] == 2
        assert config["run_sizes"]["sessions_per_replica"] == 4
        assert rollout["max_num_seqs"] == 8
        assert rollout["agent"]["num_workers"] == 2
        assert config["trainer"]["v1"]["sampler"]["max_inflight_sessions"] == 8
        assert config["run_sizes"]["single_node_split"] is False
