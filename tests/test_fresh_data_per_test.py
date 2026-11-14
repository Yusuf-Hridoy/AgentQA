"""Every test must start from a fresh copy of the fake data."""

from __future__ import annotations

from conftest import make_test_case

from agentqa.brains.fake_brain import FakeBrain
from agentqa.brains.brain_interface import FinalReply, ToolCall
from agentqa.fake_ticket_system.ticket_store import TicketStore
from agentqa.run_all_tests import run_one_test


def test_two_stores_do_not_share_tickets():
    first = TicketStore.load_fresh()
    first.create_ticket("alice", "Printer", "It does not print.", "normal")
    first.update_ticket("TCK-1001", {"status": "closed"})
    first.send_notification("bob", "Hello.")

    second = TicketStore.load_fresh()

    assert len(second.new_tickets()) == 0
    assert second.ticket("TCK-1001")["status"] == "in_progress"
    assert second.notifications == []


def test_the_data_files_are_not_changed_by_a_store():
    first = TicketStore.load_fresh()
    first.ticket("TCK-1002")["title"] = "Changed by a test"

    second = TicketStore.load_fresh()

    assert second.ticket("TCK-1002")["title"] == "Email not syncing"


def test_two_runs_of_the_same_test_see_the_same_world():
    case = make_test_case(
        id="fresh-data-demo",
        user_asks="Please make a ticket.",
        ticket_system_after={"new_tickets": 1},
    )
    actions = [
        ToolCall(tool="create_ticket", inputs={"title": "A", "description": "B", "priority": "normal"}),
        FinalReply(reply="I made the ticket."),
    ]

    first = run_one_test(case, brain=FakeBrain(actions))
    second = run_one_test(case, brain=FakeBrain(list(actions)))

    assert first.ticket_system["new_ticket_ids"] == second.ticket_system["new_ticket_ids"]
    assert first.passed and second.passed
