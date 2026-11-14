"""The shape every brain must follow.

A brain only decides the next action. It never touches the ticket system itself:
the agent loop sends every tool call through the gateway.

Phase 1 has one brain (the fake brain). Later phases add real AI brains.
They only have to follow this file.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Message(BaseModel):
    """One line of the conversation the brain can read."""

    role: str  # user, assistant, tool or system
    content: str
    tool: str | None = None


class ToolSpec(BaseModel):
    """What a brain is told about one tool it may use."""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """The brain wants to use a tool."""

    tool: str
    inputs: dict[str, Any] = Field(default_factory=dict)


class FinalReply(BaseModel):
    """The brain is done and answers the user."""

    reply: str


BrainAction = ToolCall | FinalReply


@runtime_checkable
class Brain(Protocol):
    """Every brain must have this one method."""

    def next_action(self, conversation: list[Message], tools: list[ToolSpec]) -> BrainAction:
        ...
