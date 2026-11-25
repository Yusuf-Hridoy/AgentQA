"""A scripted chat model, so CI can run both LangGraph agents with no key.

It plays back the same `fake_brain_scripts/<test id>.yaml` the fake brain plays
back, turned into the messages LangGraph expects: a model message with tool
calls, then a model message with the final reply. The conversation is ignored
on purpose, so a run is free, fast and the same every time.

The prebuilt ReAct agent binds tools to the model before it runs, so this model
has a `bind_tools` that gives itself back. It has no tools to bind: the script
already says which tool comes next.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from agentqa.brains.brain_interface import FinalReply, ToolCall
from agentqa.brains.fake_brain import NO_MORE_ACTIONS_REPLY, FakeBrain

# The fake model costs nothing, so it reports no tokens at all. A made-up
# number would turn up in the report as if it were measured.
NO_TOKENS = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _as_message(action: Any, number: int) -> AIMessage:
    """One scripted action as one model message."""
    if isinstance(action, FinalReply):
        return AIMessage(content=action.reply, usage_metadata=dict(NO_TOKENS))
    if isinstance(action, ToolCall):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": action.tool,
                    "args": dict(action.inputs),
                    "id": "call_" + str(number),
                    "type": "tool_call",
                }
            ],
            usage_metadata=dict(NO_TOKENS),
        )
    raise TypeError("A fake script holds a tool call or a reply, not " + type(action).__name__)


class FakeChatModel(BaseChatModel):
    """Plays back a fixed list of actions, one per model turn."""

    actions: list[Any] = Field(default_factory=list)
    source: str = "memory"
    turn: int = 0

    @property
    def _llm_type(self) -> str:
        return "agentqa-fake"

    @classmethod
    def from_script_file(cls, file_path: str | Path) -> "FakeChatModel":
        brain = FakeBrain.from_script_file(file_path)
        return cls(actions=list(brain.actions), source=brain.source)

    @classmethod
    def for_test_case(cls, test_id: str, folder: str | Path | None = None) -> "FakeChatModel":
        brain = FakeBrain.for_test_case(test_id, folder=folder)
        return cls(actions=list(brain.actions), source=brain.source)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeChatModel":
        """The prebuilt agent binds tools before it runs. The script needs none."""
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.turn >= len(self.actions):
            message = AIMessage(content=NO_MORE_ACTIONS_REPLY, usage_metadata=dict(NO_TOKENS))
        else:
            message = _as_message(self.actions[self.turn], self.turn + 1)
        self.turn += 1
        return ChatResult(generations=[ChatGeneration(message=message)])
