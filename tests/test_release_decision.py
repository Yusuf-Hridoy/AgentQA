"""One failed test blocks the release. Critical failures are counted apart."""

from __future__ import annotations

from conftest import make_test_case, make_trace

from agentqa.checks.check_result import AREA_SAFETY, AREA_STEPS, AREA_TASK, failed, passed
from agentqa.release_decision import BLOCKED, READY, decide_release
from agentqa.run_all_tests import TestOutcome


def make_outcome(test_id: str, checks, critical: bool = False) -> TestOutcome:
    case = make_test_case(id=test_id, critical=critical)
    return TestOutcome(
        test_case=case,
        checks=checks,
        trace=make_trace(reply="Done."),
        final_reply="Done.",
        steps_used=1,
    )


def all_good():
    return [
        passed("a", "fine", AREA_TASK),
        passed("b", "fine", AREA_STEPS),
        passed("c", "fine", AREA_SAFETY),
    ]


def one_bad_step():
    return [
        passed("a", "fine", AREA_TASK),
        failed("b", "Called create_ticket 2 times. Expected at most 1.", AREA_STEPS),
        passed("c", "fine", AREA_SAFETY),
    ]


def test_all_passed_means_ready():
    decision = decide_release([make_outcome("one", all_good()), make_outcome("two", all_good())])

    assert decision.status == READY
    assert decision.is_ready
    assert decision.exit_code() == 0
    assert decision.passed_tests == 2


def test_one_failed_test_blocks_the_release():
    decision = decide_release([make_outcome("one", all_good()), make_outcome("two", one_bad_step())])

    assert decision.status == BLOCKED
    assert decision.exit_code() == 1
    assert decision.failed_test_ids == ["two"]


def test_the_areas_are_counted_one_by_one():
    decision = decide_release([make_outcome("one", all_good()), make_outcome("two", one_bad_step())])

    assert decision.areas["task"].as_text() == "2/2"
    assert decision.areas["steps"].as_text() == "1/2"
    assert decision.areas["safety"].as_text() == "2/2"


def test_a_failure_in_a_critical_test_is_counted_as_a_critical_failure():
    decision = decide_release([make_outcome("danger", one_bad_step(), critical=True)])

    assert decision.critical_failures == 1
    assert decision.critical_failure_ids == ["danger"]


def test_a_failure_in_a_critical_test_also_counts_against_safety():
    outcome = make_outcome("danger", one_bad_step(), critical=True)

    assert outcome.area_passed()["safety"] is False
    assert decide_release([outcome]).areas["safety"].as_text() == "0/1"


def test_a_passing_critical_test_is_not_counted():
    decision = decide_release([make_outcome("safe", all_good(), critical=True)])

    assert decision.critical_failures == 0
    assert decision.status == READY


def test_the_summary_line_says_the_numbers():
    decision = decide_release([make_outcome("one", all_good()), make_outcome("two", one_bad_step())])

    assert decision.summary_line() == "Result: 1 of 2 passed - Release: BLOCKED"


def test_the_saved_shape_holds_everything_the_report_needs():
    data = decide_release([make_outcome("one", one_bad_step(), critical=True)]).to_dict()

    assert data["release"] == "BLOCKED"
    assert data["failed"] == 1
    assert data["critical_failures"] == 1
    assert data["areas"]["steps"] == {"passed": 0, "total": 1}
