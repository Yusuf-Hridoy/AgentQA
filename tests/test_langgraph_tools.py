"""The gateway tools a LangGraph agent is given.

This is the piece Phase 5 rests on. If one tool call could skip the gateway,
the trace would be a guess and every check above it would be worthless.
"""

from __future__ import annotations

import pytest

from agentqa.fake_ticket_system.ticket_store import TicketStore
from agentqa.fake_ticket_system.tool_gateway import ToolGateway
from agentqa.trace import TOOL_CALL, TOOL_ERROR, TOOL_RESULT, Trace

pytest.importorskip("langgraph")

from agentqa.agents.langgraph.langgraph_tools import (  # noqa: E402
    READ_ONLY_TOOLS,
    WRITE_TOOLS,
    StepCounter,
    build_gateway_tools,
    is_write_tool,
    tool_names_of,
)


def make_gateway(acting_user: str = "alice") -> ToolGateway:
    store = TicketStore.load_fresh()
    trace = Trace(test_id="demo-case", acting_user=acting_user)
    return ToolGateway(store=store, acting_user=acting_user, trace=trace)


def test_an_employee_is_given_only_the_tools_an_employee_may_use():
    gateway = make_gateway("alice")
    tools = build_gateway_tools(gateway, "employee", StepCounter())

    assert tool_names_of(tools) == [
        "search_help_docs",
        "get_ticket",
        "create_ticket",
        "update_ticket",
    ]
    assert "assign_ticket" not in tool_names_of(tools)
    assert "send_notification" not in tool_names_of(tools)


def test_an_it_agent_is_given_every_tool():
    gateway = make_gateway("maria")
    tools = build_gateway_tools(gateway, "it_agent", StepCounter())
    assert len(tools) == 6
    assert "assign_ticket" in tool_names_of(tools)


def test_every_call_goes_through_the_gateway_and_into_the_trace():
    gateway = make_gateway("alice")
    tools = {one.name: one for one in build_gateway_tools(gateway, "employee", StepCounter())}

    tools["search_help_docs"].invoke({"query": "printer"})

    kinds = [event.kind for event in gateway.trace.events]
    assert kinds == [TOOL_CALL, TOOL_RESULT]
    assert gateway.trace.events[0].tool == "search_help_docs"


def test_the_ticket_store_only_changes_through_the_gateway():
    gateway = make_gateway("alice")
    tools = {one.name: one for one in build_gateway_tools(gateway, "employee", StepCounter())}
    before = len(gateway.store.snapshot()["new_ticket_ids"])

    tools["create_ticket"].invoke(
        {"title": "Printer", "description": "It does not print.", "priority": "normal"}
    )

    after = gateway.store.snapshot()["new_ticket_ids"]
    assert len(after) == before + 1
    assert gateway.trace.successful_calls("create_ticket")


def test_the_step_numbers_count_up_one_per_tool_call():
    gateway = make_gateway("alice")
    counter = StepCounter()
    tools = {one.name: one for one in build_gateway_tools(gateway, "employee", counter)}

    tools["search_help_docs"].invoke({"query": "printer"})
    tools["search_help_docs"].invoke({"query": "vpn"})
    tools["create_ticket"].invoke({"title": "A", "description": "B"})

    steps = [event.step for event in gateway.trace.events if event.kind == TOOL_CALL]
    assert steps == [1, 2, 3]
    assert counter.peek() == 4


def test_bad_input_reaches_the_gateway_so_the_trace_shows_the_mistake():
    """A LangChain tool would refuse this itself and the trace would stay empty.

    The builtin agent sends the call, the gateway refuses it and writes a
    tool_error. Both agents have to land in the same place, or a comparison
    between them is measuring the wrapper and not the agent.
    """
    gateway = make_gateway("alice")
    tools = {one.name: one for one in build_gateway_tools(gateway, "employee", StepCounter())}

    answer = tools["create_ticket"].invoke({"title": "No description here"})

    assert gateway.trace.failed_calls("create_ticket")
    assert gateway.trace.events[-1].kind == TOOL_ERROR
    assert "description" in str(answer)


def test_a_call_the_user_may_not_make_is_blocked_by_the_gateway():
    """alice is an employee. The tool is not even built for her, and if the
    agent asks for it anyway the gateway still says no."""
    gateway = make_gateway("alice")
    counter = StepCounter()
    build_gateway_tools(gateway, "employee", counter)

    from agentqa.agents.langgraph.langgraph_tools import call_the_gateway

    said = call_the_gateway(gateway, counter, "assign_ticket", {"ticket_id": "TCK-1001", "assignee": "bob"})

    assert gateway.trace.blocked_calls("assign_ticket")
    assert "Not allowed" in said


def test_the_tools_are_split_into_reading_and_writing():
    assert set(READ_ONLY_TOOLS) == {"search_help_docs", "get_ticket"}
    for name in WRITE_TOOLS:
        assert is_write_tool(name)
    for name in READ_ONLY_TOOLS:
        assert not is_write_tool(name)


def test_the_model_is_shown_the_same_fields_the_builtin_brain_is_shown():
    gateway = make_gateway("alice")
    tools = {one.name: one for one in build_gateway_tools(gateway, "employee", StepCounter())}

    fields = tools["create_ticket"].args
    assert sorted(fields) == ["description", "priority", "title"]
    assert fields["priority"]["enum"] == ["low", "normal", "high"]
