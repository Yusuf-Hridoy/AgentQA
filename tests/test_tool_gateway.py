"""The gateway is the only door to the tools, so it gets its own tests."""

from __future__ import annotations

from agentqa.fake_ticket_system.ticket_store import TicketStore
from agentqa.fake_ticket_system.tool_gateway import FORCED_FAILURE_MESSAGE, ToolGateway
from agentqa.trace import BLOCKED, TOOL_CALL, TOOL_ERROR, TOOL_RESULT, Trace

GOOD_TICKET = {"title": "Printer", "description": "It does not print.", "priority": "normal"}


def build_gateway(acting_user="alice", tool_failures=None):
    store = TicketStore.load_fresh()
    trace = Trace(test_id="gateway-test", acting_user=acting_user)
    return store, trace, ToolGateway(store, acting_user, trace, tool_failures)


def test_a_good_call_runs_and_is_recorded():
    store, trace, gateway = build_gateway()

    outcome = gateway.call("create_ticket", GOOD_TICKET, step=1)

    assert outcome.ok
    assert len(store.new_tickets()) == 1
    assert [event.kind for event in trace.events] == [TOOL_CALL, TOOL_RESULT]


def test_a_forbidden_tool_is_blocked_and_recorded():
    store, trace, gateway = build_gateway()

    outcome = gateway.call("assign_ticket", {"ticket_id": "TCK-1001", "assignee": "bob"}, step=1)

    assert outcome.kind == BLOCKED
    assert "Not allowed" in str(outcome.error)
    assert trace.blocked_calls()[0].tool == "assign_ticket"
    assert store.ticket("TCK-1001")["assignee"] == "maria"


def test_another_users_ticket_is_blocked():
    store, trace, gateway = build_gateway()

    outcome = gateway.call("get_ticket", {"ticket_id": "TCK-1003"}, step=1)

    assert outcome.kind == BLOCKED
    assert "belongs to another user" in str(outcome.error)


def test_an_it_agent_may_use_every_tool():
    store, trace, gateway = build_gateway(acting_user="maria")

    assert gateway.call("get_ticket", {"ticket_id": "TCK-1003"}, step=1).ok
    assert gateway.call("send_notification", {"to": "bob", "message": "Hello."}, step=2).ok
    assert len(store.notifications) == 1


def test_bad_input_is_refused_before_the_tool_runs():
    store, trace, gateway = build_gateway()

    outcome = gateway.call("create_ticket", {"title": "Printer"}, step=1)

    assert outcome.kind == TOOL_ERROR
    assert "Wrong input for create_ticket" in str(outcome.error)
    assert store.new_tickets() == []


def test_an_unknown_field_is_refused():
    store, trace, gateway = build_gateway()

    outcome = gateway.call("create_ticket", dict(GOOD_TICKET, owner="bob"), step=1)

    assert outcome.kind == TOOL_ERROR
    assert store.new_tickets() == []


def test_an_unknown_tool_is_refused():
    store, trace, gateway = build_gateway()

    outcome = gateway.call("delete_everything", {}, step=1)

    assert outcome.kind == TOOL_ERROR
    assert "no tool called" in str(outcome.error)


def test_a_forced_failure_fails_only_the_first_calls():
    store, trace, gateway = build_gateway(tool_failures={"create_ticket": 2})

    first = gateway.call("create_ticket", GOOD_TICKET, step=1)
    second = gateway.call("create_ticket", GOOD_TICKET, step=2)
    third = gateway.call("create_ticket", GOOD_TICKET, step=3)

    assert first.error == FORCED_FAILURE_MESSAGE
    assert second.error == FORCED_FAILURE_MESSAGE
    assert third.ok
    assert len(store.new_tickets()) == 1


def test_update_ticket_cannot_change_the_assignee():
    store, trace, gateway = build_gateway(acting_user="maria")

    outcome = gateway.call("update_ticket", {"ticket_id": "TCK-1001", "fields": {"assignee": "bob"}}, step=1)

    assert outcome.kind == TOOL_ERROR
    assert store.ticket("TCK-1001")["assignee"] == "maria"


def test_a_missing_ticket_is_a_tool_error_not_a_permission_problem():
    store, trace, gateway = build_gateway()

    outcome = gateway.call("get_ticket", {"ticket_id": "TCK-9999"}, step=1)

    assert outcome.kind == TOOL_ERROR
    assert "No ticket with id" in str(outcome.error)


def test_every_call_is_recorded_even_when_it_fails():
    store, trace, gateway = build_gateway(tool_failures={"create_ticket": 1})

    gateway.call("create_ticket", GOOD_TICKET, step=1)
    gateway.call("assign_ticket", {"ticket_id": "TCK-1001", "assignee": "bob"}, step=2)
    gateway.call("get_ticket", {"ticket_id": "TCK-1001"}, step=3)

    assert len(trace.attempted_calls()) == 3
    assert len(trace.failed_calls()) == 1
    assert len(trace.blocked_calls()) == 1
    assert len(trace.successful_calls()) == 1
