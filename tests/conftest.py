"""Small helpers for the AgentQA unit tests.

These tests check AgentQA itself, not the agent. Most of them build a trace by
hand, so a check can be tested without running an agent at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentqa.test_case_schema import TestCase
from agentqa.trace import (
    BLOCKED,
    FINAL_REPLY,
    STEP_LIMIT_HIT,
    TOOL_CALL,
    TOOL_ERROR,
    TOOL_RESULT,
    Trace,
)


def make_test_case(**overrides: Any) -> TestCase:
    """A plain test case. Pass only the fields a test cares about."""
    data: dict[str, Any] = {
        "id": "demo-case",
        "type": "normal",
        "acting_user": "alice",
        "user_asks": "Please help me.",
    }
    data.update(overrides)
    return TestCase(**data)


def ok_call(tool: str, inputs: dict[str, Any] | None = None) -> tuple[str, dict[str, Any], str]:
    return (tool, inputs or {}, TOOL_RESULT)


def blocked_call(tool: str, inputs: dict[str, Any] | None = None) -> tuple[str, dict[str, Any], str]:
    return (tool, inputs or {}, BLOCKED)


def failed_call(tool: str, inputs: dict[str, Any] | None = None) -> tuple[str, dict[str, Any], str]:
    return (tool, inputs or {}, TOOL_ERROR)


def make_trace(
    calls: list[tuple[str, dict[str, Any], str]] | None = None,
    reply: str | None = None,
    hit_step_limit: bool = False,
) -> Trace:
    """Build a trace by hand, the way a real run would write it."""
    trace = Trace(test_id="demo-case", acting_user="alice")
    step = 0
    for tool, inputs, kind in calls or []:
        step += 1
        trace.add(step=step, kind=TOOL_CALL, tool=tool, inputs=inputs)
        if kind == TOOL_RESULT:
            trace.add(step=step, kind=TOOL_RESULT, tool=tool, inputs=inputs, output={"ok": True})
        elif kind == BLOCKED:
            trace.add(step=step, kind=BLOCKED, tool=tool, inputs=inputs, error="Not allowed: test says so.")
        else:
            trace.add(step=step, kind=TOOL_ERROR, tool=tool, inputs=inputs, error="Service down.")
    if reply is not None:
        trace.add(step=step + 1, kind=FINAL_REPLY, output=reply)
    if hit_step_limit:
        trace.add(step=step, kind=STEP_LIMIT_HIT, error="Out of steps.")
    return trace


@pytest.fixture
def fresh_store():
    from agentqa.fake_ticket_system.ticket_store import TicketStore

    return TicketStore.load_fresh()
