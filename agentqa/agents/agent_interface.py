"""The shape every agent under test follows.

An agent is given one test case, one gateway and the role of the user it is
helping. It decides what to do. It may only touch the ticket system through
the gateway, so the trace and the permission checks are the same whichever
agent ran.

The three agents:

    builtin           AgentQA's own loop, unchanged, and still the default
    langgraph-react   a standard LangGraph ReAct agent
    langgraph-guard   a LangGraph agent with a guard node before every write

The checks, the claims and the judge run after this, and they do not know or
care which agent produced the trace.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentqa.agent_loop import AgentRunResult
from agentqa.brains.brain_interface import TokenUsage
from agentqa.fake_ticket_system.tool_gateway import ToolGateway
from agentqa.test_case_schema import TestCase

AGENT_BUILTIN = "builtin"
AGENT_LANGGRAPH_REACT = "langgraph-react"
AGENT_LANGGRAPH_GUARD = "langgraph-guard"

ALL_AGENT_NAMES = (AGENT_BUILTIN, AGENT_LANGGRAPH_REACT, AGENT_LANGGRAPH_GUARD)
LANGGRAPH_AGENT_NAMES = (AGENT_LANGGRAPH_REACT, AGENT_LANGGRAPH_GUARD)

# Only the guard agent has a guard. Everything else reports no guard at all.
NO_FRAMEWORK = "none"


class AgentSetupError(Exception):
    """The agent cannot be used at all. AgentQA stops with exit code 2."""


class GuardBlock(BaseModel):
    """One action a guard refused before it reached the gateway."""

    tool: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "inputs": dict(self.inputs), "reason": self.reason}


class GuardDecision(BaseModel):
    """What a guard decided about one proposed action, and why."""

    allow: bool
    reason: str = ""


def is_langgraph_agent(agent_name: str) -> bool:
    return agent_name in LANGGRAPH_AGENT_NAMES


@runtime_checkable
class AgentUnderTest(Protocol):
    """Every agent must have a name and this one method."""

    name: str

    def run(
        self,
        test_case: TestCase,
        gateway: ToolGateway,
        role: str,
        max_steps: int,
    ) -> AgentRunResult:
        ...


def agent_usage(agent: Any) -> TokenUsage:
    """Tokens the agent itself used. Zero when it does not count them."""
    totals = getattr(agent, "usage_totals", None)
    if totals is None:
        return TokenUsage()
    return totals()


def guard_usage(agent: Any) -> TokenUsage:
    """Tokens the guard used. Zero for every agent that has no guard."""
    totals = getattr(agent, "guard_usage_totals", None)
    if totals is None:
        return TokenUsage()
    return totals()
