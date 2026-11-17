from pear.agent_loop.tool_parser import JsonBlockToolParser
from verl.experimental.agent_loop.tool_parser import ToolParser


@ToolParser.register("qwen3_5")
class Qwen3_5ToolParser(JsonBlockToolParser):
    """PEAR's JSON action after the last </think>, for Qwen3.5's multimodal models."""

    THINK_END = "</think>"


@ToolParser.register("qwen3_8")
class Qwen3_8ToolParser(Qwen3_5ToolParser):
    """The existing Qwen3.8 format name, using the same action protocol."""
