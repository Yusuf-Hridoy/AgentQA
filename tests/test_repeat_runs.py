"""Running one test more than once, and what the runs together mean.

One run cannot show whether an agent is reliable. These tests pin down the
four labels and the strict rule that goes with them: a test passed only when
every run of it passed.
"""

from __future__ import annotations

import json

import pytest
from conftest import make_test_case, make_trace

from agentqa.checks.check_result import (
    AREA_HONESTY,
    AREA_SAFETY,
    AREA_STEPS,
    AREA_TASK,
    failed,
    passed,
)
from agentqa.checks.claim_checks import CLAIM_CHECK_NAME
from agentqa.release_decision import BLOCKED, READY, decide_release
from agentqa.repeat_runs import (
    ERROR,
    FLAKY,
    MAX_REPEAT,
    STABLE_FAIL,
    STABLE_PASS,
    RepeatCountError,
    RepeatedTest,
    settled_repeat,
    stability_of,
    trace_path_for,
)
from agentqa.run_all_tests import TestOutcome


def all_good():
    return [
        passed("a", "fine", AREA_TASK),
        passed("b", "fine", AREA_STEPS),
        passed("c", "fine", AREA_SAFETY),
        passed(CLAIM_CHECK_NAME, "fine", AREA_HONESTY),
    ]


def one_lie():
    return [
        passed("a", "fine", AREA_TASK),
        passed("b", "fine", AREA_STEPS),
        passed("c", "fine", AREA_SAFETY),
        failed(CLAIM_CHECK_NAME, "Says it did create_ticket, but it never did.", AREA_HONESTY),
    ]


def one_bad_step():
    return [
        passed("a", "fine", AREA_TASK),
        failed("right_steps", "Never used create_ticket.", AREA_STEPS),
        passed("c", "fine", AREA_SAFETY),
        passed(CLAIM_CHECK_NAME, "fine", AREA_HONESTY),
    ]


def a_run(checks, error: str | None = None, test_id: str = "demo-case") -> TestOutcome:
    return TestOutcome(
        test_case=make_test_case(id=test_id),
        checks=checks,
        trace=make_trace(reply="Done."),
        final_reply="Done.",
        steps_used=1,
        error_message=error,
    )


def repeated(*run_checks, test_id: str = "demo-case") -> RepeatedTest:
    return RepeatedTest(
        test_case=make_test_case(id=test_id),
        runs=[a_run(checks, test_id=test_id) for checks in run_checks],
    )


# ---------- the four labels ----------


def test_three_passes_is_a_stable_pass():
    test = repeated(all_good(), all_good(), all_good())

    assert test.stability == STABLE_PASS
    assert test.runs_text == "3/3"
    assert test.passed
    assert not test.is_flaky


def test_two_passes_and_one_fail_is_flaky():
    test = repeated(all_good(), one_lie(), all_good())

    assert test.stability == FLAKY
    assert test.runs_text == "2/3"
    assert test.is_flaky
    assert not test.passed


def test_no_passes_is_a_stable_fail():
    test = repeated(one_lie(), one_lie(), one_lie())

    assert test.stability == STABLE_FAIL
    assert test.runs_text == "0/3"
    assert not test.passed


def test_an_error_with_no_failure_is_an_error():
    test = RepeatedTest(
        test_case=make_test_case(id="demo-case"),
        runs=[a_run(all_good()), a_run([], error="The model did not answer.")],
    )

    assert test.stability == ERROR
    assert test.is_error
    assert not test.passed


def test_an_error_next_to_a_real_failure_is_not_called_an_error():
    """A test that plainly failed is a failure. The error is still counted."""
    test = RepeatedTest(
        test_case=make_test_case(id="demo-case"),
        runs=[a_run(one_lie()), a_run([], error="The model did not answer.")],
    )

    assert test.stability == STABLE_FAIL
    assert not test.is_error


def test_a_label_for_no_runs_at_all():
    assert stability_of([]) == ERROR


# ---------- the strict rule ----------


def test_one_failed_run_of_one_test_blocks_the_release():
    good = repeated(all_good(), all_good(), all_good(), test_id="good")
    flaky = repeated(all_good(), all_good(), one_lie(), test_id="sometimes-lies")

    decision = decide_release([good, flaky], repeat=3)

    assert decision.status == BLOCKED
    assert decision.passed_tests == 1
    assert decision.flaky == 1
    assert decision.flaky_test_ids == ["sometimes-lies"]
    assert decision.passed_text() == "1 of 2 passed every run"


def test_every_run_passing_is_ready():
    decision = decide_release(
        [repeated(all_good(), all_good(), test_id="one"), repeated(all_good(), all_good(), test_id="two")],
        repeat=2,
    )

    assert decision.status == READY
    assert decision.flaky == 0


def test_an_area_counts_only_when_it_passed_in_every_run():
    test = repeated(all_good(), one_bad_step(), all_good())

    areas = test.area_passed()
    assert areas["steps"] is False
    assert areas["task"] is True
    assert areas["safety"] is True
    assert areas["honesty"] is True


def test_a_critical_failure_in_any_run_counts_for_the_test():
    test = repeated(all_good(), all_good(), one_lie())

    assert test.is_critical_failure
    assert decide_release([test], repeat=3).critical_failures == 1


def test_the_first_failed_run_is_the_one_shown():
    test = repeated(all_good(), one_bad_step(), one_lie())

    assert test.first_failed_run_number == 2
    assert test.main_area == "steps"
    assert test.main_message == "Never used create_ticket."


# ---------- the repeat number ----------


def test_the_default_repeat_is_one():
    assert settled_repeat(None) == 1
    assert settled_repeat(1) == 1


def test_five_runs_are_allowed():
    assert settled_repeat(MAX_REPEAT) == MAX_REPEAT


@pytest.mark.parametrize("asked", [0, -1, 6, 99])
def test_a_repeat_outside_the_range_is_refused_with_a_clear_message(asked):
    with pytest.raises(RepeatCountError) as problem:
        settled_repeat(asked)

    said = str(problem.value)
    assert "--repeat" in said
    assert "1 to 5" in said
    assert str(asked) in said


# ---------- where the traces go ----------


def test_one_run_keeps_the_old_trace_layout():
    assert trace_path_for("printer-ticket", 1, 1) == "traces/printer-ticket.json"


def test_more_than_one_run_gives_each_test_a_folder():
    assert trace_path_for("printer-ticket", 2, 3) == "traces/printer-ticket/run-2.json"


def test_the_saved_test_keeps_the_fields_it_had_before():
    test = repeated(all_good(), one_lie())

    as_dict = test.to_dict()
    for field_name in ("id", "type", "user_asks", "checks", "areas", "final_reply", "trace_file"):
        assert field_name in as_dict, field_name
    assert as_dict["stability"] == FLAKY
    assert as_dict["runs_passed"] == 1
    assert as_dict["runs_total"] == 2
    assert [run["run"] for run in as_dict["runs"]] == [1, 2]
    assert as_dict["runs"][1]["trace_file"] == "traces/demo-case/run-2.json"


# ---------- what a real run writes ----------


def test_one_run_writes_the_traces_where_they_have_always_been(tmp_path):
    """--repeat 1 must leave the saved layout exactly as it was."""
    from typer.testing import CliRunner

    from agentqa.cli import app
    from agentqa.results_writer import RESULTS_FILE_NAME

    result = CliRunner().invoke(app, ["run", "--brain", "fake", "--reports-dir", str(tmp_path)])
    assert result.exit_code == 0

    folder = sorted(tmp_path.glob("run-*"))[-1]
    traces = folder / "traces"
    assert (traces / "printer-ticket.json").is_file()
    assert not (traces / "printer-ticket").exists()

    saved = json.loads((folder / RESULTS_FILE_NAME).read_text(encoding="utf-8"))
    printer = [test for test in saved["tests"] if test["id"] == "printer-ticket"][0]
    assert printer["trace_file"] == "traces/printer-ticket.json"
    assert printer["status"] == "pass"
    assert printer["stability"] == STABLE_PASS
    assert printer["runs_total"] == 1
    assert saved["schema_version"] == 4
    assert saved["summary"]["repeat"] == 1
    assert saved["run_info"]["repeat"] == 1


def test_more_than_one_run_writes_a_trace_per_run(tmp_path):
    from typer.testing import CliRunner

    from agentqa.cli import app
    from agentqa.results_writer import RESULTS_FILE_NAME

    result = CliRunner().invoke(
        app, ["run", "--brain", "fake", "--repeat", "2", "--reports-dir", str(tmp_path)]
    )
    assert result.exit_code == 0

    folder = sorted(tmp_path.glob("run-*"))[-1]
    assert (folder / "traces" / "printer-ticket" / "run-1.json").is_file()
    assert (folder / "traces" / "printer-ticket" / "run-2.json").is_file()
    assert not (folder / "traces" / "printer-ticket.json").exists()

    saved = json.loads((folder / RESULTS_FILE_NAME).read_text(encoding="utf-8"))
    assert saved["summary"]["repeat"] == 2
    assert "Flaky: 0" in result.output
    assert "10 of 10 passed every run" in result.output


def test_the_command_refuses_a_repeat_outside_the_range(tmp_path):
    from typer.testing import CliRunner

    from agentqa.cli import app

    result = CliRunner().invoke(
        app, ["run", "--brain", "fake", "--repeat", "6", "--reports-dir", str(tmp_path)]
    )

    assert result.exit_code == 2
    assert "1 to 5" in result.output
    assert not list(tmp_path.glob("run-*"))
