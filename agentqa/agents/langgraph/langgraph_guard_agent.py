"""A LangGraph agent with a guard node in front of every write.

The graph:

    agent -> guard -> tools -> agent

and it ends when the agent answers without calling a tool. Reading tools
(search_help_docs, get_ticket) walk past the guard. Every write
(create_ticket, update_ticket, assign_ticket, send_notification) is checked one
at a time before it runs.

What the guard sees, and this is the whole point:

    the user's own request, and
    the proposed action: the tool name and its inputs.

It never sees a tool result, a ticket, a help article or the agent's
conversation. Text planted inside a ticket cannot reach it, because the ticket
is never put in front of it. `guard_question` in guard_prompt.py builds the
whole of the guard's input, and it takes exactly those two things.

A blocked action never reaches the gateway, so the ticket system never changed
and the gateway-based checks see nothing: the system did not do it. The guard's
work is never hidden, though. Every block is written into the trace as
GUARD_BLOCKED and counted, split into what it stopped and what it wrongly
blocked. And if the agent then says it did the blocked thing, claim_vs_reality
still fails the test. Honesty is checked whatever the guard did.
"""

from __future__ import annotations

import json
from typing import Any

import openai
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph

from agentqa.agent_loop import AgentRunResult
from agentqa.agents.agent_interface import (
    AGENT_LANGGRAPH_GUARD,
    GuardBlock,
    GuardDecision,
)
from agentqa.agents.langgraph.guard_prompt import (
    ALLOW_KEY,
    GUARD_SYSTEM_PROMPT,
    REASON_KEY,
    guard_question,
)
from agentqa.agents.langgraph.langgraph_model import (
    UsageCounter,
    as_brain_error,
    reply_text_of,
)
from agentqa.agents.langgraph.langgraph_react_agent import (
    last_reply_message,
    record_reply,
    system_prompt_for,
)
from agentqa.agents.langgraph.langgraph_tools import (
    StepCounter,
    build_gateway_tools,
    call_the_gateway,
    is_write_tool,
)
from agentqa.brains.brain_interface import TokenUsage
from agentqa.fake_ticket_system.tool_gateway import ToolGateway
from agentqa.test_case_schema import TestCase
from agentqa.trace import GUARD_BLOCKED, STEP_LIMIT_HIT

BLOCKED_MESSAGE_START = "Blocked by the guard: the user did not ask for this. Reason: "
COULD_NOT_DECIDE = "Guard could not decide, so the action was blocked."


def recursion_limit_for(max_steps: int) -> int:
    """How many nodes LangGraph may run for this many AgentQA steps.

    One step here is three nodes, not two: the model, the guard, then the
    tools. The run ends on one more model node that answers instead of calling
    a tool. The ReAct agent has no guard node, so its number is smaller.
    """
    return 3 * int(max_steps) + 1


def blocked_tool_message(reason: str, call_id: str) -> ToolMessage:
    """What the agent is told when the guard refuses one of its actions."""
    return ToolMessage(content=BLOCKED_MESSAGE_START + str(reason), tool_call_id=call_id)


def decision_from_text(text: str) -> GuardDecision | None:
    """Read the guard's JSON answer. None when it cannot be read at all."""
    found = _first_object(str(text or ""))
    if found is None:
        return None
    try:
        data = json.loads(found)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or ALLOW_KEY not in data:
        return None
    reason = str(data.get(REASON_KEY) or "").strip()
    return GuardDecision(allow=bool(data[ALLOW_KEY]), reason=reason)


def _first_object(text: str) -> str | None:
    """The first {...} in the answer, so a fenced or chatty reply still reads."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    return text[start : end + 1]


class ModelGuard:
    """A guard that asks a model, with its own token count.

    It is the same model the agent thinks with, on purpose. The question this
    phase asks is whether the graph can protect a weak model, not whether a
    stronger model can.
    """

    name = "model"

    def __init__(self, model: Any) -> None:
        self.model = model
        self.usage = UsageCounter()

    def usage_totals(self) -> TokenUsage:
        return self.usage.totals()

    def decide(self, user_asks: str, tool_name: str, inputs: dict) -> GuardDecision:
        question = guard_question(user_asks, tool_name, inputs)
        messages = [SystemMessage(content=GUARD_SYSTEM_PROMPT), HumanMessage(content=question)]

        for _try in range(2):
            answer = self.model.invoke(messages, config={"callbacks": [self.usage]})
            decision = decision_from_text(reply_text_of(answer))
            if decision is not None:
                return decision

        # Fail closed. An answer we cannot read is not permission.
        return GuardDecision(allow=False, reason=COULD_NOT_DECIDE)


class GuardState(MessagesState):
    """The messages, plus the calls this turn's guard let through."""

    allowed_calls: list[dict]


class LangGraphGuardAgent:
    """A LangGraph agent whose every write is checked before it runs."""

    name = AGENT_LANGGRAPH_GUARD

    def __init__(self, model: Any, guard: Any) -> None:
        self.model = model
        self.guard = guard
        self.usage = UsageCounter()

    def usage_totals(self) -> TokenUsage:
        return self.usage.totals()

    def guard_usage_totals(self) -> TokenUsage:
        """The guard's own tokens, counted apart from the agent's."""
        totals = getattr(self.guard, "usage_totals", None)
        if totals is None:
            return TokenUsage()
        return totals()

    # ---------- the graph ----------

    def _build_graph(
        self,
        test_case: TestCase,
        gateway: ToolGateway,
        role: str,
        counter: StepCounter,
        tools: list[Any],
        blocks: list[GuardBlock],
    ) -> Any:
        model_with_tools = self.model.bind_tools(tools)
        system = SystemMessage(content=system_prompt_for(test_case, role))
        tools_by_name = {str(one.name): one for one in tools}

        def agent_node(state: dict) -> dict:
            answer = model_with_tools.invoke([system] + list(state["messages"]))
            return {"messages": [answer], "allowed_calls": []}

        def guard_node(state: dict) -> dict:
            last = state["messages"][-1]
            allowed: list[dict] = []
            refused: list[ToolMessage] = []

            for call in list(getattr(last, "tool_calls", None) or []):
                name = str(call.get("name") or "")
                inputs = dict(call.get("args") or {})

                # A reading tool changes nothing, so there is nothing to allow.
                if not is_write_tool(name):
                    allowed.append(call)
                    continue

                decision = self.guard.decide(test_case.user_asks, name, inputs)
                if decision.allow:
                    allowed.append(call)
                    continue

                # The action stops here. It never reaches the gateway.
                blocks.append(GuardBlock(tool=name, inputs=inputs, reason=decision.reason))
                gateway.trace.add(
                    step=counter.take(),
                    kind=GUARD_BLOCKED,
                    tool=name,
                    inputs=inputs,
                    error=decision.reason,
                )
                refused.append(blocked_tool_message(decision.reason, str(call.get("id") or "")))

            return {"messages": refused, "allowed_calls": allowed}

        def tools_node(state: dict) -> dict:
            said = []
            for call in list(state.get("allowed_calls") or []):
                name = str(call.get("name") or "")
                inputs = dict(call.get("args") or {})
                tool = tools_by_name.get(name)
                if tool is None:
                    # A tool this role may not use. The gateway says so and
                    # writes it into the trace, as it does for every call.
                    text = call_the_gateway(gateway, counter, name, inputs)
                else:
                    text = str(tool.invoke(inputs))
                said.append(ToolMessage(content=text, tool_call_id=str(call.get("id") or "")))
            return {"messages": said, "allowed_calls": []}

        def after_agent(state: dict) -> str:
            last = state["messages"][-1]
            return "guard" if getattr(last, "tool_calls", None) else END

        def after_guard(state: dict) -> str:
            # Everything blocked means there is nothing to run, so the agent
            # is asked again with the guard's refusals in front of it.
            return "tools" if state.get("allowed_calls") else "agent"

        graph = StateGraph(GuardState)
        graph.add_node("agent", agent_node)
        graph.add_node("guard", guard_node)
        graph.add_node("tools", tools_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", after_agent, {"guard": "guard", END: END})
        graph.add_conditional_edges("guard", after_guard, {"tools": "tools", "agent": "agent"})
        graph.add_edge("tools", "agent")
        return graph.compile()

    # ---------- one run ----------

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
        blocks: list[GuardBlock] = []
        graph = self._build_graph(test_case, gateway, role, counter, tools, blocks)

        try:
            state = graph.invoke(
                {"messages": [HumanMessage(content=test_case.user_asks)], "allowed_calls": []},
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
                guard_blocks=list(blocks),
            )
        except openai.APIError as problem:
            raise as_brain_error(problem) from problem
        except (ConnectionError, TimeoutError, OSError) as problem:
            raise as_brain_error(problem) from problem

        steps_used = counter.peek()
        reply = record_reply(trace, steps_used, last_reply_message(state.get("messages")))

        return AgentRunResult(
            trace=trace,
            final_reply=reply,
            step_limit_hit=False,
            steps_used=steps_used,
            guard_blocks=list(blocks),
        )


__all__ = [
    "BLOCKED_MESSAGE_START",
    "COULD_NOT_DECIDE",
    "GuardState",
    "LangGraphGuardAgent",
    "ModelGuard",
    "blocked_tool_message",
    "decision_from_text",
    "recursion_limit_for",
]
