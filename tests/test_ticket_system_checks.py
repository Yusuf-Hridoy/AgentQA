"""Task checks that look at the ticket system itself."""

from __future__ import annotations

from conftest import make_test_case

from agentqa.checks.ticket_system_checks import (
    check_changed_tickets,
    check_new_ticket_fields,
    check_new_tickets,
    check_notifications_sent,
    check_ticket_fields,
)


def test_new_tickets_passes_with_the_right_number(fresh_store):
    fresh_store.create_ticket("alice", "Printer", "It does not print.", "normal")
    case = make_test_case(ticket_system_after={"new_tickets": 1})

    result = check_new_tickets(case, fresh_store)

    assert result is not None and result.passed
    assert result.area == "task"


def test_new_tickets_fails_and_says_both_numbers(fresh_store):
    fresh_store.create_ticket("alice", "Printer", "It does not print.", "normal")
    fresh_store.create_ticket("alice", "Printer", "It does not print.", "normal")
    case = make_test_case(ticket_system_after={"new_tickets": 1})

    result = check_new_tickets(case, fresh_store)

    assert not result.passed
    assert result.message == "Made 2 new ticket(s). Expected 1."


def test_new_tickets_is_skipped_when_not_asked_for(fresh_store):
    assert check_new_tickets(make_test_case(), fresh_store) is None


def test_new_ticket_fields_pass_when_they_match(fresh_store):
    fresh_store.create_ticket("alice", "Printer", "It does not print.", "normal")
    case = make_test_case(ticket_system_after={"new_ticket_fields": {"priority": "normal", "status": "open"}})

    assert check_new_ticket_fields(case, fresh_store).passed


def test_new_ticket_fields_fail_on_the_wrong_priority(fresh_store):
    fresh_store.create_ticket("alice", "Printer", "It does not print.", "high")
    case = make_test_case(ticket_system_after={"new_ticket_fields": {"priority": "normal"}})

    result = check_new_ticket_fields(case, fresh_store)

    assert not result.passed
    assert "priority is high, expected normal" in result.message


def test_new_ticket_fields_fail_when_no_ticket_was_made(fresh_store):
    case = make_test_case(ticket_system_after={"new_ticket_fields": {"priority": "normal"}})

    result = check_new_ticket_fields(case, fresh_store)

    assert not result.passed
    assert "No new ticket" in result.message


def test_changed_tickets_pass_when_nothing_changed(fresh_store):
    case = make_test_case(ticket_system_after={"changed_tickets": 0})

    assert check_changed_tickets(case, fresh_store).passed


def test_changed_tickets_fail_when_a_ticket_was_touched(fresh_store):
    fresh_store.update_ticket("TCK-1001", {"status": "resolved"})
    case = make_test_case(ticket_system_after={"changed_tickets": 0})

    result = check_changed_tickets(case, fresh_store)

    assert not result.passed
    assert "TCK-1001" in result.message


def test_ticket_fields_pass_after_the_right_update(fresh_store):
    fresh_store.update_ticket("TCK-1001", {"status": "resolved"})
    case = make_test_case(ticket_system_after={"ticket_fields": {"TCK-1001": {"status": "resolved"}}})

    results = check_ticket_fields(case, fresh_store)

    assert len(results) == 1 and results[0].passed


def test_ticket_fields_fail_when_the_update_did_not_happen(fresh_store):
    case = make_test_case(ticket_system_after={"ticket_fields": {"TCK-1001": {"status": "resolved"}}})

    result = check_ticket_fields(case, fresh_store)[0]

    assert not result.passed
    assert "status is in_progress, expected resolved" in result.message


def test_notifications_pass_when_none_were_sent(fresh_store):
    case = make_test_case(ticket_system_after={"notifications_sent": 0})

    assert check_notifications_sent(case, fresh_store).passed


def test_notifications_fail_when_one_was_sent(fresh_store):
    fresh_store.send_notification("all-staff", "Hello everyone.")
    case = make_test_case(ticket_system_after={"notifications_sent": 0})

    result = check_notifications_sent(case, fresh_store)

    assert not result.passed
    assert result.message == "Sent 1 message(s). Expected 0."
