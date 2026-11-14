"""The trace: everything the agent did in one test, in order.

A person must be able to open the saved trace and see exactly what happened.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# The kinds of events a trace can hold.
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
BLOCKED = "blocked"
TOOL_ERROR = "tool_error"
FINAL_REPLY = "final_reply"
STEP_LIMIT_HIT = "step_limit_hit"


class TraceEvent(BaseModel):
    step: int
    kind: str
    tool: str | None = None
    inputs: dict[str, Any] | None = None
    output: Any | None = None
    error: str | None = None
    time_ms: float = 0.0


class Trace(BaseModel):
    """All events of one test run, plus small helpers the checks use."""

    test_id: str = ""
    acting_user: str = ""
    events: list[TraceEvent] = Field(default_factory=list)

    def add(self, **event_fields: Any) -> TraceEvent:
        event = TraceEvent(**event_fields)
        self.events.append(event)
        return event

    # ---------- questions the checks ask ----------

    def events_of_kind(self, kind: str) -> list[TraceEvent]:
        return [e for e in self.events if e.kind == kind]

    def attempted_calls(self, tool: str | None = None) -> list[TraceEvent]:
        """Every call the agent tried, even blocked or failed ones."""
        calls = self.events_of_kind(TOOL_CALL)
        return [e for e in calls if tool is None or e.tool == tool]

    def successful_calls(self, tool: str | None = None) -> list[TraceEvent]:
        results = self.events_of_kind(TOOL_RESULT)
        return [e for e in results if tool is None or e.tool == tool]

    def blocked_calls(self, tool: str | None = None) -> list[TraceEvent]:
        blocked = self.events_of_kind(BLOCKED)
        return [e for e in blocked if tool is None or e.tool == tool]

    def failed_calls(self, tool: str | None = None) -> list[TraceEvent]:
        errors = self.events_of_kind(TOOL_ERROR)
        return [e for e in errors if tool is None or e.tool == tool]

    def called_tool_names_in_order(self) -> list[str]:
        return [e.tool or "" for e in self.attempted_calls()]

    def count_calls(self, tool: str) -> int:
        return len(self.attempted_calls(tool))

    def final_reply(self) -> str | None:
        replies = self.events_of_kind(FINAL_REPLY)
        return replies[-1].output if replies else None

    def hit_step_limit(self) -> bool:
        return bool(self.events_of_kind(STEP_LIMIT_HIT))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
