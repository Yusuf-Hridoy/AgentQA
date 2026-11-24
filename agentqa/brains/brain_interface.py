"""The shape every brain must follow.

A brain only decides the next action. It never touches the ticket system itself:
the agent loop sends every tool call through the gateway.

Phase 2 has two kinds of brain: the fake brain (a script) and the real brain
(an OpenAI-compatible model). Both only have to follow this file.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Message(BaseModel):
    """One line of the conversation the brain can read."""

    role: str  # user, assistant, tool or system
    content: str
    tool: str | None = None
    # Set on an assistant tool call and on the tool result that answers it, so a
    # real model can be shown the conversation in its own format.
    tool_call_id: str | None = None
    tool_inputs: dict[str, Any] | None = None
    # Anything the provider attached to its own tool call and wants back when
    # the call is replayed. Gemini puts a thought_signature here. Most
    # providers put nothing, and then this stays empty.
    tool_extra: dict[str, Any] | None = None


class ToolSpec(BaseModel):
    """What a brain is told about one tool it may use."""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """The brain wants to use a tool."""

    tool: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    # The id the model gave this call. The fake brain leaves it empty.
    call_id: str | None = None
    # Opaque data the provider attached and wants back. Never read, only
    # carried. See Message.tool_extra.
    provider_extra: dict[str, Any] | None = None


class FinalReply(BaseModel):
    """The brain is done and answers the user.

    An empty answer is reported as one: reply stays empty and empty_reply is
    true. We never put words in the agent's mouth, because the checks and the
    judge would then grade text the agent never wrote.
    """

    reply: str
    # Why the provider stopped: stop, length, content_filter, tool_calls ...
    # Empty for a brain that has no provider, such as the fake brain.
    finish_reason: str | None = None
    # True when the provider gave no usable answer text at all.
    empty_reply: bool = False


BrainAction = ToolCall | FinalReply


class TokenUsage(BaseModel):
    """How much one brain or judge used. Always counted, never guessed."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    time_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt_tokens: int, completion_tokens: int, time_ms: float) -> None:
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)
        self.calls += 1
        self.time_ms = round(self.time_ms + float(time_ms or 0.0), 3)

    def plus(self, other: "TokenUsage") -> "TokenUsage":
        """Both together. Used when a run has a judge and a claim reader."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            calls=self.calls + other.calls,
            time_ms=round(self.time_ms + other.time_ms, 3),
        )

    def minus(self, earlier: "TokenUsage") -> "TokenUsage":
        """What was used since the earlier snapshot."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens - earlier.prompt_tokens,
            completion_tokens=self.completion_tokens - earlier.completion_tokens,
            calls=self.calls - earlier.calls,
            time_ms=round(self.time_ms - earlier.time_ms, 3),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }


@runtime_checkable
class Brain(Protocol):
    """Every brain must have this one method."""

    def next_action(self, conversation: list[Message], tools: list[ToolSpec]) -> BrainAction:
        ...
