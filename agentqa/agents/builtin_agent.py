"""AgentQA's own agent loop, behind the agent interface.

Nothing about the loop changes here. This file only gives it the shape every
agent has, so `run_one_test` can ask for any agent and treat them all alike.
A builtin run must give exactly the same results it gave before Phase 5.
"""

from __future__ import annotations

from agentqa.agent_loop import AgentRunResult, run_agent_loop, tools_the_user_may_use
from agentqa.agents.agent_interface import AGENT_BUILTIN
from agentqa.brains.brain_interface import Brain, TokenUsage
from agentqa.fake_ticket_system.tool_gateway import ToolGateway
from agentqa.test_case_schema import TestCase


class BuiltinAgent:
    """Today's loop: one brain, one tool call per step, no framework."""

    name = AGENT_BUILTIN

    def __init__(self, brain: Brain) -> None:
        self.brain = brain

    def usage_totals(self) -> TokenUsage:
        """The brain's own tokens. The loop itself costs nothing."""
        totals = getattr(self.brain, "usage_totals", None)
        if totals is None:
            return TokenUsage()
        return totals()

    def run(
        self,
        test_case: TestCase,
        gateway: ToolGateway,
        role: str,
        max_steps: int,
    ) -> AgentRunResult:
        # The loop reads max_steps off the test case itself, which is where it
        # has always come from. role is worked out there too, from the store.
        return run_agent_loop(
            brain=self.brain,
            gateway=gateway,
            test_case=test_case,
            tools=tools_the_user_may_use(role),
        )
