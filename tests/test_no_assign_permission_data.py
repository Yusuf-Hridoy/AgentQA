"""The permission test must ask for something that is not already done.

TCK-1001 is already assigned to maria, so "assign TCK-1001 to maria" let a
truthful agent answer "it is already assigned" and fail a test about
permission. The case now points at a ticket with no assignee.
"""

from __future__ import annotations

from agentqa.fake_ticket_system.ticket_store import TicketStore
from agentqa.test_case_loader import load_test_cases

CASE_ID = "no-assign-permission"
WANTED_ASSIGNEE = "maria"


def the_case():
    cases = load_test_cases(only=CASE_ID)
    assert len(cases) == 1
    return cases[0]


def ticket_id_in(text: str) -> str:
    words = [word.strip(".,:;\"'") for word in text.split()]
    ids = [word for word in words if word.startswith("TCK-")]
    assert len(ids) == 1, "The case must name exactly one ticket: " + text
    return ids[0]


def test_the_case_asks_about_a_ticket_that_has_no_assignee():
    case = the_case()
    store = TicketStore.load_fresh()

    ticket = store.ticket(ticket_id_in(case.user_asks))

    assert ticket["assignee"] is None
    assert ticket["owner"] == case.acting_user
    assert ticket["status"] == "open"


def test_the_asked_for_assignee_is_not_already_on_the_ticket():
    case = the_case()
    store = TicketStore.load_fresh()

    ticket = store.ticket(ticket_id_in(case.user_asks))

    assert ticket["assignee"] != WANTED_ASSIGNEE
    assert WANTED_ASSIGNEE in case.user_asks


def test_tck_1001_is_untouched_because_other_tests_use_it():
    store = TicketStore.load_fresh()

    ticket = store.ticket("TCK-1001")

    assert ticket["status"] == "in_progress"
    assert ticket["assignee"] == "maria"
