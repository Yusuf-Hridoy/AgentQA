"""What really changed in the ticket system?

These checks look at the store itself, not at what the agent said it did.
"""

from __future__ import annotations

from typing import Any

from agentqa.checks.check_result import AREA_TASK, CheckResult, failed, passed
from agentqa.fake_ticket_system.ticket_store import TicketStore
from agentqa.test_case_schema import TestCase


def check_new_tickets(test_case: TestCase, store: TicketStore) -> CheckResult | None:
    wanted = test_case.ticket_system_after.new_tickets
    if wanted is None:
        return None
    made = len(store.new_tickets())
    name_of_check = "new_tickets"
    if made == wanted:
        return passed(name_of_check, "Made " + str(made) + " new ticket(s), as expected.", AREA_TASK)
    return failed(
        name_of_check,
        "Made " + str(made) + " new ticket(s). Expected " + str(wanted) + ".",
        AREA_TASK,
    )


def check_new_ticket_fields(test_case: TestCase, store: TicketStore) -> CheckResult | None:
    wanted: dict[str, Any] = test_case.ticket_system_after.new_ticket_fields
    if not wanted:
        return None

    name_of_check = "new_ticket_fields"
    new_tickets = store.new_tickets()
    if not new_tickets:
        return failed(name_of_check, "No new ticket was made, so its fields could not be checked.", AREA_TASK)

    ticket = new_tickets[0]
    wrong = []
    for field, value in wanted.items():
        found = ticket.get(field)
        if found != value:
            wrong.append(field + " is " + str(found) + ", expected " + str(value))
    if wrong:
        return failed(
            name_of_check,
            "The new ticket " + str(ticket.get("ticket_id")) + " is wrong: " + "; ".join(wrong) + ".",
            AREA_TASK,
        )
    return passed(
        name_of_check,
        "The new ticket " + str(ticket.get("ticket_id")) + " has the expected fields.",
        AREA_TASK,
    )


def check_changed_tickets(test_case: TestCase, store: TicketStore) -> CheckResult | None:
    wanted = test_case.ticket_system_after.changed_tickets
    if wanted is None:
        return None
    changed = store.changed_ticket_ids()
    name_of_check = "changed_tickets"
    if len(changed) == wanted:
        return passed(
            name_of_check,
            "Changed " + str(len(changed)) + " old ticket(s), as expected.",
            AREA_TASK,
        )
    return failed(
        name_of_check,
        "Changed " + str(len(changed)) + " old ticket(s) (" + (", ".join(changed) or "none")
        + "). Expected " + str(wanted) + ".",
        AREA_TASK,
    )


def check_ticket_fields(test_case: TestCase, store: TicketStore) -> list[CheckResult]:
    results: list[CheckResult] = []
    for ticket_id, wanted in test_case.ticket_system_after.ticket_fields.items():
        name_of_check = "ticket_fields_" + ticket_id
        ticket = store.ticket_or_none(ticket_id)
        if ticket is None:
            results.append(failed(name_of_check, "There is no ticket " + ticket_id + ".", AREA_TASK))
            continue
        wrong = []
        for field, value in wanted.items():
            found = ticket.get(field)
            if found != value:
                wrong.append(field + " is " + str(found) + ", expected " + str(value))
        if wrong:
            results.append(
                failed(name_of_check, "Ticket " + ticket_id + " is wrong: " + "; ".join(wrong) + ".", AREA_TASK)
            )
        else:
            results.append(passed(name_of_check, "Ticket " + ticket_id + " looks as expected.", AREA_TASK))
    return results


def check_notifications_sent(test_case: TestCase, store: TicketStore) -> CheckResult | None:
    wanted = test_case.ticket_system_after.notifications_sent
    if wanted is None:
        return None
    sent = len(store.notifications)
    name_of_check = "notifications_sent"
    if sent == wanted:
        return passed(name_of_check, "Sent " + str(sent) + " message(s), as expected.", AREA_TASK)
    return failed(
        name_of_check,
        "Sent " + str(sent) + " message(s). Expected " + str(wanted) + ".",
        AREA_TASK,
    )


def run_ticket_system_checks(test_case: TestCase, store: TicketStore) -> list[CheckResult]:
    results: list[CheckResult] = []
    for one_check in (check_new_tickets, check_new_ticket_fields, check_changed_tickets, check_notifications_sent):
        result = one_check(test_case, store)
        if result is not None:
            results.append(result)
    results.extend(check_ticket_fields(test_case, store))
    return results
