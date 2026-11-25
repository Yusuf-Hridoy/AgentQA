"""A standard LangGraph ReAct agent, tested by AgentQA.

Nothing here is special to AgentQA except the tools, and the tools are the
point: they are the gateway. The agent itself is the prebuilt one, built the
way the LangGraph documentation builds it.

The system message is exactly the text the builtin agent gets. It is taken from
`starting_conversation`, so AGENT_RULES is never copied or reworded here and
`agent_rules_hash` keeps meaning what it says.

The step limit. LangGraph counts every node it runs, and a ReAct turn is two
nodes: the model, then the tools. So `max_steps` turns into a recursion limit
of 2 * max_steps + 1: two nodes per step, plus the last model call that answers
without a tool call. Hitting it is recorded as STEP_LIMIT_HIT with no final
reply, which is what the builtin loop records.
"""

from __future__ import annotations

from typing import Any

import openai
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from agentqa.agent_loop import AgentRunResult, starting_conversation
from agentqa.agents.agent_interface import AGENT_LANGGRAPH_REACT
from agentqa.agents.langgraph.langgraph_model import (
    UsageCounter,
    as_brain_error,
    finish_reason_of,
    reply_text_of,
)
from agentqa.agents.langgraph.langgraph_tools import StepCounter, build_gateway_tools
from agentqa.brains.brain_interface import TokenUsage
from agentqa.fake_ticket_system.tool_gateway import ToolGateway
from agentqa.test_case_schema import TestCase
from agentqa.trace import FINAL_REPLY, STEP_LIMIT_HIT


def recursion_limit_for(max_steps: int) -> int:
    """How many nodes LangGraph may run for this many AgentQA steps.

    One step is a model node and a tool node, and the run ends on one more
    model node that answers instead of calling a tool.
    """
    return 2 * int(max_steps) + 1


def system_prompt_for(test_case: TestCase, role: str) -> str:
    """The same system message the builtin agent is given, word for word."""
    return starting_conversation(test_case, role)[0].content


def last_reply_message(messages: list[Any]) -> Any | None:
    """The last model message that answered instead of calling a tool."""
    for message in reversed(list(messages or [])):
        if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            return message
    return None


def record_reply(
    trace: Any,
    step: int,
    message: Any | None,
) -> str:
    """Write the final reply into the trace, honestly.

    An empty answer is recorded as empty. We never put words in the agent's
    mouth: the checks and the judge would then grade text it never wrote.
    """
    reply = reply_text_of(message) if message is not None else ""
    trace.add(
        step=step,
        kind=FINAL_REPLY,
        output=reply,
        finish_reason=finish_reason_of(message) if message is not None else None,
        empty_reply=not reply,
    )
    return reply


class LangGraphReactAgent:
    """The prebuilt LangGraph ReAct agent, with AgentQA's gateway tools."""

    name = AGENT_LANGGRAPH_REACT

    def __init__(self, model: Any) -> None:
        self.model = model
        self.usage = UsageCounter()

    def usage_totals(self) -> TokenUsage:
        return self.usage.totals()

    def run(
        self,
        test_case: TestCase,
        gateway: ToolGateway,
        role: str,
        max_steps: int,
    ) -> AgentRunResult:
        trace = gateway.trace
        counter = StepCounter()
        tools = build_gateway_tools(gateway, role, counter)

        agent = create_agent(
            model=self.model,
            tools=tools,
            system_prompt=system_prompt_for(test_case, role),
        )

        try:
            state = agent.invoke(
                {"messages": [HumanMessage(content=test_case.user_asks)]},
                config={
                    "recursion_limit": recursion_limit_for(max_steps),
                    "callbacks": [self.usage],
                },
            )
        except GraphRecursionError:
            trace.add(
                step=max_steps,
                kind=STEP_LIMIT_HIT,
                error="The agent used all " + str(max_steps) + " steps without an answer.",
            )
            return AgentRunResult(
                trace=trace,
                final_reply=None,
                step_limit_hit=True,
                steps_used=max_steps,
            )
        except openai.APIError as problem:
            raise as_brain_error(problem) from problem
        except (ConnectionError, TimeoutError, OSError) as problem:
            raise as_brain_error(problem) from problem

        # One step per tool call, then one more for the turn that answered.
        steps_used = counter.peek()
        reply = record_reply(trace, steps_used, last_reply_message(state.get("messages")))

        return AgentRunResult(
            trace=trace,
            final_reply=reply,
            step_limit_hit=False,
            steps_used=steps_used,
        )
