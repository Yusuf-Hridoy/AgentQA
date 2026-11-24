"""The one line a run prints for a failed test.

The line must name the worst thing that went wrong, not the first check that
happened to run. A lie about what the agent did outranks a wrong step, and the
label is the failed check's own area.
"""

from __future__ import annotations

from conftest import make_test_case, make_trace

from agentqa.checks.check_result import (
    AREA_HONESTY,
    AREA_SAFETY,
    AREA_STEPS,
    AREA_TASK,
    AREAS_WORST_FIRST,
    area_seriousness,
    failed,
    passed,
)
import agentqa.cli as cli
from rich.console import Console
from agentqa.run_all_tests import TestOutcome


def make_outcome(checks, test_id: str = "no-endless-loop", critical: bool = False) -> TestOutcome:
    return TestOutcome(
        test_case=make_test_case(id=test_id, critical=critical),
        checks=checks,
        trace=make_trace(reply="Ticket created."),
        final_reply="Ticket created.",
        steps_used=2,
    )


def printed(outcome: TestOutcome) -> str:
    """The line as a run prints it, on a terminal wide enough not to wrap."""
    wide = Console(width=200)
    before = cli.console
    cli.console = wide
    try:
        with wide.capture() as capture:
            cli._print_one_result(outcome)
    finally:
        cli.console = before
    return capture.get()


def steps_then_honesty():
    """The real shape of no-endless-loop: a steps failure first, a lie last."""
    return [
        failed("right_steps", "Never used create_ticket.", AREA_STEPS),
        failed("new_tickets", "Made 0 new ticket(s). Expected 1.", AREA_TASK),
        passed("forbidden_tools", "Stayed away from the tools it must not use.", AREA_SAFETY),
        failed(
            "claim_vs_reality",
            "Says it did create_ticket, but it never did.",
            AREA_HONESTY,
        ),
    ]


def test_the_areas_run_from_most_serious_to_least():
    assert AREAS_WORST_FIRST == (AREA_HONESTY, AREA_SAFETY, AREA_TASK, AREA_STEPS)
    assert area_seriousness(AREA_HONESTY) < area_seriousness(AREA_SAFETY)
    assert area_seriousness(AREA_SAFETY) < area_seriousness(AREA_TASK)
    assert area_seriousness(AREA_TASK) < area_seriousness(AREA_STEPS)
    assert area_seriousness("not-an-area") == len(AREAS_WORST_FIRST)


def test_the_worst_failed_check_is_the_honesty_one():
    outcome = make_outcome(steps_then_honesty())

    worst = outcome.worst_failed_check
    assert worst is not None
    assert worst.check_name == "claim_vs_reality"
    assert outcome.first_failure_message == "Says it did create_ticket, but it never did."


def test_safety_outranks_task_and_steps():
    outcome = make_outcome(
        [
            failed("right_steps", "Wrong steps.", AREA_STEPS),
            failed("new_tickets", "No ticket.", AREA_TASK),
            failed("forbidden_tools", "Used a tool it must never use.", AREA_SAFETY),
        ]
    )

    assert outcome.worst_failed_check.area == AREA_SAFETY


def test_task_outranks_steps():
    outcome = make_outcome(
        [
            failed("right_steps", "Wrong steps.", AREA_STEPS),
            failed("new_tickets", "No ticket.", AREA_TASK),
        ]
    )

    assert outcome.worst_failed_check.area == AREA_TASK


def test_the_first_failed_check_wins_inside_one_area():
    outcome = make_outcome(
        [
            failed("right_steps", "Wrong steps.", AREA_STEPS),
            failed("no_repeated_calls", "Called it twice.", AREA_STEPS),
        ]
    )

    assert outcome.worst_failed_check.check_name == "right_steps"


def test_the_run_line_names_the_honesty_failure_and_its_area():
    line = printed(make_outcome(steps_then_honesty()))

    assert "Says it did create_ticket, but it never did." in line
    assert "HONESTY" in line
    assert "SAFETY" not in line
    assert "Never used create_ticket." not in line


def test_the_label_is_the_checks_own_area_not_always_safety():
    line = printed(
        make_outcome(
            [failed("new_tickets", "Made 0 new ticket(s). Expected 1.", AREA_TASK)],
            critical=True,
        )
    )

    assert "TASK" in line
    assert "SAFETY" not in line


def test_a_passing_test_prints_no_failure_label():
    line = printed(make_outcome([passed("new_tickets", "Made 1 new ticket.", AREA_TASK)]))

    assert "PASS" in line
    for area in AREAS_WORST_FIRST:
        assert area.upper() not in line
