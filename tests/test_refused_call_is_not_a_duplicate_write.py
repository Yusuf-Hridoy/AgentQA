"""A call the system refused is not a second write.

Three Groq tests (create-ticket-vpn, no-endless-loop, notify-needs-consent) were
reported as doing the same write twice. They did not. The model sent a value the
store does not accept ("High", "Medium", "Resolved"), the store refused the call
without changing anything, the model corrected itself and the write happened
once. Two things were wrong on our side, and both are tested here:

1. The tool never told the model which values it accepts, so the model had to
   guess. That cost a step on every run.
2. max_calls counted the refused try, so one ticket looked like two.
"""

from __future__ import annotations

from typing import Any

from conftest import failed_call, make_test_case, make_trace, ok_call

from agentqa.agent_loop import run_agent_loop
from agentqa.brains.brain_interface import BrainAction, FinalReply, Message, ToolCall, ToolSpec
from agentqa.checks.loop_checks import check_retry_limit
from agentqa.checks.tool_call_checks import check_max_calls, check_no_repeated_calls
from agentqa.fake_ticket_system.it_support_tools import tool_specs
from agentqa.fake_ticket_system.ticket_store import ALLOWED_PRIORITY, ALLOWED_STATUS, TicketStore
from agentqa.fake_ticket_system.tool_gateway import ToolGateway
from agentqa.trace import Trace


class ScriptedBrain:
    """Plays a fixed list of actions, one per step."""

    def __init__(self, actions: list[BrainAction]) -> None:
        self.actions = list(actions)
        self.seen: list[list[Message]] = []

    def next_action(self, conversation: list[Message], tools: list[ToolSpec]) -> BrainAction:
        self.seen.append(list(conversation))
        return self.actions.pop(0)


def _schema_of(tool_name: str) -> dict[str, Any]:
    return tool_specs([tool_name])[0].input_schema


# ---------- 1. the tool says which values it accepts ----------


def test_create_ticket_tells_the_model_the_priorities_it_accepts():
    """Without this the model guesses "High" or "Medium" and wastes a call."""
    priority = _schema_of("create_ticket")["properties"]["priority"]

    assert priority["enum"] == list(ALLOWED_PRIORITY)


def test_update_ticket_tells_the_model_the_values_it_accepts():
    spec = tool_specs(["update_ticket"])[0]
    fields = spec.input_schema["properties"]["fields"]
    said = spec.description + " " + fields["description"]

    for value in ALLOWED_STATUS + ALLOWED_PRIORITY:
        assert value in said


# ---------- 2. a refused call is not counted as work done ----------


def test_a_corrected_retry_is_one_write_not_two():
    case = make_test_case(max_calls={"update_ticket": 1})
    trace = make_trace(
        [
            failed_call("update_ticket", {"ticket_id": "TCK-1001", "fields": {"status": "Resolved"}}),
            ok_call("update_ticket", {"ticket_id": "TCK-1001", "fields": {"status": "resolved"}}),
        ],
        reply="TCK-1001 is resolved.",
    )

    assert check_max_calls(case, trace)[0].passed
    assert check_no_repeated_calls(case, trace).passed


def test_hammering_a_tool_with_bad_input_still_fails_the_retry_limit():
    """max_calls lets the refused tries go, so this is what has to catch them."""
    case = make_test_case(max_calls={"create_ticket": 1})
    trace = make_trace([failed_call("create_ticket")] * 4, reply="Sorry.")

    results = check_retry_limit(case, trace)

    assert len(results) == 1
    assert not results[0].passed
    assert "4 times" in results[0].message


# ---------- the whole way through: loop, gateway, store, checks ----------


def test_the_groq_run_end_to_end_makes_one_ticket_and_passes_the_limit():
    """The exact shape of the Groq trace: a refused call, then a good one."""
    case = make_test_case(
        acting_user="alice",
        max_calls={"create_ticket": 1},
        max_steps=6,
    )
    store = TicketStore.load_fresh()
    trace = Trace(test_id=case.id, acting_user=case.acting_user)
    gateway = ToolGateway(store=store, acting_user=case.acting_user, trace=trace)
    brain = ScriptedBrain(
        [
            ToolCall(
                tool="create_ticket",
                inputs={"title": "VPN", "description": "No VPN from home.", "priority": "High"},
                call_id="call_a",
            ),
            ToolCall(
                tool="create_ticket",
                inputs={"title": "VPN", "description": "No VPN from home.", "priority": "high"},
                call_id="call_b",
            ),
            FinalReply(reply="I made a ticket for your VPN problem."),
        ]
    )

    run_agent_loop(brain=brain, gateway=gateway, test_case=case)

    assert len(store.new_tickets()) == 1
    assert check_max_calls(case, trace)[0].passed
    assert [e.kind for e in trace.events] == [
        "tool_call", "tool_error", "tool_call", "tool_result", "final_reply",
    ]


def test_the_agent_is_told_what_the_system_refused():
    """The refusal has to reach the model, or it cannot correct itself."""
    case = make_test_case(acting_user="alice", max_steps=4)
    store = TicketStore.load_fresh()
    trace = Trace(test_id=case.id, acting_user=case.acting_user)
    gateway = ToolGateway(store=store, acting_user=case.acting_user, trace=trace)
    brain = ScriptedBrain(
        [
            ToolCall(
                tool="update_ticket",
                inputs={"ticket_id": "TCK-1001", "fields": {"status": "Resolved"}},
                call_id="call_a",
            ),
            FinalReply(reply="I could not set that status."),
        ]
    )

    run_agent_loop(brain=brain, gateway=gateway, test_case=case)

    answer = brain.seen[-1][-1]
    assert answer.role == "tool"
    assert answer.tool_call_id == "call_a"
    assert "Status must be one of" in answer.content
