import asyncio
import json
import logging
import types
from unittest.mock import AsyncMock

import pytest

from pear.agent_loop.agent_loop import PearAgentLoop
from pear.agent_loop.episode import EndReason, Episode
from pear.agent_loop.tool_parser import NoToolCall
from pear.envs.libero.env import LiberoEnv
from pear.envs.tool import EnvironmentTool
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

ENV_ARGS = {"suite": "libero_goal", "task_id": 3, "trial_id": 0}
ZERO = {"arm": {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}, "gripper": {"width": 0.0}}
# json blocks that are no action of LIBERO's schema, each with the tool's reason
INVALID_ACTIONS = [
    ([ZERO], "the json block must map exactly the parts arm, gripper to their quantities"),
    ({"arm": ZERO["arm"]}, "the json block must map exactly the parts arm, gripper to their quantities"),
    ({**ZERO, "gripper": {}}, "'gripper' must give exactly its quantities width"),
    ({**ZERO, "arm": 1.0}, "'arm' must give exactly its quantities x, y, z, roll, pitch, yaw"),
    ({**ZERO, "gripper": {"width": float("nan")}}, "gripper.width must be a finite number"),
    ({**ZERO, "arm": {**ZERO["arm"], "yaw": "1"}}, "arm.yaw must be a finite number"),
]


class UnreachedClient:
    """A client of an episode whose simulator no call may reach."""

    def __init__(self, env_class: type) -> None:
        self.env_class = env_class

    async def step(self, uid: str, action) -> dict:
        raise AssertionError(f"the action {action} reached the simulator")


@pytest.fixture
def loop(tmp_path, monkeypatch) -> PearAgentLoop:
    """A PearAgentLoop in an open episode of LIBERO's tool, without tokenizer or server."""
    monkeypatch.setattr(LiberoEnv, "CLIENT_CLASS_PATH", f"{__name__}.{UnreachedClient.__qualname__}")
    loop = object.__new__(PearAgentLoop)
    loop.tool = EnvironmentTool({"env_class": "pear.envs.libero.env.LiberoEnv", "log_dir": str(tmp_path)})
    loop.episode = Episode(ENV_ARGS)
    loop.end_reason = None
    loop.max_action_retries = 0
    loop.action_retries = 0

    return loop


@pytest.mark.parametrize(("action", "reason"), INVALID_ACTIONS)
def test_an_invalid_action_ends_the_episode_as_a_parse_failure_with_the_tool_s_reason(
    loop, tmp_path, monkeypatch, caplog, action, reason
):
    # the loop's logger keeps its records to itself: the test's capture in the place of its handler
    monkeypatch.setattr(logging.getLogger("pear.agent_loop.agent_loop"), "handlers", [caplog.handler])
    agent_data = types.SimpleNamespace(metrics={}, tool_calls=[types.SimpleNamespace(arguments=json.dumps(action))])

    state = asyncio.run(loop._handle_processing_tools_state(agent_data))

    assert (state, loop.end_reason) == (AgentState.TERMINATED, EndReason.PARSE)
    assert caplog.messages == [f"PARSE_FAILURE env={ENV_ARGS} error={reason!r}"]
    # the answer is no action of the episode: nothing of it is accounted, and the call log has no line of it
    episode = loop.episode
    assert (episode.num_chunks, episode.actions, episode.chunk_rewards) == (0, [], [])
    assert (episode.done, episode.success) == (False, False)
    assert list(agent_data.metrics) == ["tool_calls"]
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def retry_turn(loop):
    loop.max_action_retries = 2
    loop.response_budget = 16
    data = types.SimpleNamespace(
        metrics={}, messages=[{"role": "assistant", "content": "invalid action"}], prompt_ids=[9, 1],
        response_mask=[1], response_logprobs=[-0.5], response_topk_ids=None, user_turns=1,
        tool_calls=[types.SimpleNamespace(arguments='{"arm": {}}')], response_ids=[1],
    )

    async def merge(previous, updated, tokens, mask, logprobs, tools):
        return types.SimpleNamespace(token_ids=tokens + [2]), mask + [0], logprobs + [0.0]

    loop.ct_merge_context_msg = merge
    return data


def test_missing_fields_get_feedback_and_a_bounded_retry_on_the_same_observation(loop, retry_turn):
    data = retry_turn
    for _ in range(2):
        data.tool_calls = [types.SimpleNamespace(arguments='{"arm": {}}')]
        assert asyncio.run(loop._handle_processing_tools_state(data)) == AgentState.GENERATING
        assert data.messages[-1]["role"] == "user"
        assert "gripper" in data.messages[-1]["content"] and "width" in data.messages[-1]["content"]
        assert not data.tool_calls
    data.tool_calls = [types.SimpleNamespace(arguments='{"arm": {}}')]
    assert asyncio.run(loop._handle_processing_tools_state(data)) == AgentState.TERMINATED
    assert loop.end_reason == EndReason.PARSE
    assert data.response_mask == [1, 0, 0]
    assert loop.episode.num_chunks == 0 and loop.episode.actions == []


def test_a_corrected_action_executes_once_and_resets_retries(loop, retry_turn, monkeypatch):
    data = retry_turn
    assert asyncio.run(loop._handle_processing_tools_state(data)) == AgentState.GENERATING
    data.tool_calls = [types.SimpleNamespace(arguments=json.dumps(ZERO))]
    client_step = AsyncMock(return_value=("observation", {"actions": [ZERO], "rewards": [0], "done": False}))
    monkeypatch.setattr(loop.tool, "step", client_step)
    assert asyncio.run(loop._handle_processing_tools_state(data)) == AgentState.GENERATING
    client_step.assert_awaited_once_with(loop.episode, ZERO)
    assert loop.episode.actions == [ZERO] and loop.action_retries == 0


def test_feedback_must_fit_the_existing_episode_budget(loop, retry_turn):
    loop.response_budget = 2
    assert asyncio.run(loop._handle_processing_tools_state(retry_turn)) == AgentState.TERMINATED
    assert loop.end_reason == EndReason.BUDGET
    assert retry_turn.response_mask == [1] and loop.episode.num_chunks == 0


def test_missing_json_can_retry_through_the_generation_path(loop, retry_turn, monkeypatch):
    monkeypatch.setenv("PEAR_LOG_TURNS", "0")
    monkeypatch.setattr(ToolAgentLoop, "_handle_generating_state", AsyncMock(return_value=AgentState.TERMINATED))
    retry_turn.tool_calls = []
    loop.tool_parser = types.SimpleNamespace(parse=AsyncMock(side_effect=NoToolCall("no JSON", "text", "text")))
    assert asyncio.run(loop._handle_generating_state(retry_turn, {})) == AgentState.GENERATING
    assert "no JSON" in retry_turn.messages[-1]["content"] and loop.episode.num_chunks == 0
