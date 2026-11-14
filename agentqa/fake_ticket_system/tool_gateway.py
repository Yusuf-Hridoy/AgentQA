"""The one door to the tools.

Every tool call from the agent goes through here. The gateway checks the input,
checks the permission, applies any forced failure the test asks for, runs the
tool and records everything in the trace. The agent cannot skip it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from agentqa.fake_ticket_system.it_support_tools import TOOLS
from agentqa.fake_ticket_system.ticket_store import TicketStore, TicketStoreError
from agentqa.fake_ticket_system.tool_permissions import check_permission
from agentqa.trace import BLOCKED, TOOL_CALL, TOOL_ERROR, TOOL_RESULT, Trace

FORCED_FAILURE_MESSAGE = "Service down. Please try again later."


@dataclass
class ToolOutcome:
    """What the gateway gives back to the agent loop."""

    kind: str  # tool_result, blocked or tool_error
    tool: str
    output: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.kind == TOOL_RESULT

    def as_text(self) -> str:
        """Short text for the conversation the brain reads."""
        if self.ok:
            return str(self.output)
        return str(self.error)


def _readable_input_error(tool_name: str, error: ValidationError) -> str:
    parts = []
    for problem in error.errors():
        where = ".".join(str(piece) for piece in problem["loc"]) or "input"
        parts.append(where + ": " + problem["msg"])
    return "Wrong input for " + tool_name + ". " + "; ".join(parts)


class ToolGateway:
    """Checks, runs and records every tool call."""

    def __init__(
        self,
        store: TicketStore,
        acting_user: str,
        trace: Trace,
        tool_failures: dict[str, int] | None = None,
    ) -> None:
        self.store = store
        self.acting_user = acting_user
        self.trace = trace
        self.failures_left: dict[str, int] = dict(tool_failures or {})

    def call(self, tool_name: str, inputs: dict[str, Any], step: int) -> ToolOutcome:
        started = time.perf_counter()
        inputs = dict(inputs or {})
        self.trace.add(step=step, kind=TOOL_CALL, tool=tool_name, inputs=inputs)

        tool = TOOLS.get(tool_name)
        if tool is None:
            return self._error(step, tool_name, inputs, "There is no tool called " + tool_name + ".", started)

        try:
            data = tool.input_model(**inputs)
        except ValidationError as problem:
            return self._error(step, tool_name, inputs, _readable_input_error(tool_name, problem), started)

        decision = check_permission(self.store, self.acting_user, tool_name, inputs)
        if not decision.allowed:
            self.trace.add(
                step=step,
                kind=BLOCKED,
                tool=tool_name,
                inputs=inputs,
                error=decision.reason,
                time_ms=self._ms(started),
            )
            return ToolOutcome(kind=BLOCKED, tool=tool_name, error=decision.reason)

        if self.failures_left.get(tool_name, 0) > 0:
            self.failures_left[tool_name] -= 1
            return self._error(step, tool_name, inputs, FORCED_FAILURE_MESSAGE, started)

        try:
            output = tool.run(self.store, self.acting_user, data)
        except TicketStoreError as problem:
            return self._error(step, tool_name, inputs, str(problem), started)

        self.trace.add(
            step=step,
            kind=TOOL_RESULT,
            tool=tool_name,
            inputs=inputs,
            output=output,
            time_ms=self._ms(started),
        )
        return ToolOutcome(kind=TOOL_RESULT, tool=tool_name, output=output)

    # ---------- small helpers ----------

    @staticmethod
    def _ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000, 3)

    def _error(self, step: int, tool_name: str, inputs: dict[str, Any], message: str, started: float) -> ToolOutcome:
        self.trace.add(
            step=step,
            kind=TOOL_ERROR,
            tool=tool_name,
            inputs=inputs,
            error=message,
            time_ms=self._ms(started),
        )
        return ToolOutcome(kind=TOOL_ERROR, tool=tool_name, error=message)
