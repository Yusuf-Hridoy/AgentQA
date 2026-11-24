"""One test, run more than once.

One run cannot show how reliable an agent is. A real Groq run lied in
`tool-down-retry` once and told the truth the next time; a real Qwen 4B run
failed `no-assign-permission` once and passed it the next time. A single green
run hides that. Repeating each test shows it.

The rule here is strict on purpose:

    a test passed      only when every run passed
    an area passed     only when it passed in every run
    a critical failure counts when any run had one

A `RepeatedTest` answers the same questions a single `TestOutcome` answers, so
`decide_release` works on either without knowing which it was given.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentqa.brains.brain_interface import TokenUsage
from agentqa.checks.check_result import ALL_AREAS
from agentqa.run_all_tests import STATUS_ERROR, STATUS_PASS, TestOutcome
from agentqa.test_case_schema import TestCase

# How many times one test may be run.
MIN_REPEAT = 1
MAX_REPEAT = 5
DEFAULT_REPEAT = 1

# What the runs together say about one test.
STABLE_PASS = "stable_pass"
FLAKY = "flaky"
STABLE_FAIL = "stable_fail"
ERROR = "error"

STABILITY_WORD = {
    STABLE_PASS: "PASS",
    FLAKY: "FLAKY",
    STABLE_FAIL: "FAIL",
    ERROR: "ERROR",
}

STABILITY_COLOUR = {
    STABLE_PASS: "green",
    FLAKY: "dark_orange",
    STABLE_FAIL: "red",
    ERROR: "yellow",
}

# For the report and the scorecard, where the word is read by a person.
STABILITY_LABEL = {
    STABLE_PASS: "Stable pass",
    FLAKY: "Flaky",
    STABLE_FAIL: "Fail",
    ERROR: "Error",
}


class RepeatCountError(ValueError):
    """The --repeat number is outside the range AgentQA allows."""


def settled_repeat(repeat: int | None) -> int:
    """Check the --repeat number before the first test runs, not after."""
    number = DEFAULT_REPEAT if repeat is None else int(repeat)
    if number < MIN_REPEAT or number > MAX_REPEAT:
        raise RepeatCountError(
            "--repeat must be a whole number from " + str(MIN_REPEAT) + " to "
            + str(MAX_REPEAT) + ". You asked for " + str(number) + "."
        )
    return number


def stability_of(runs: list[TestOutcome]) -> str:
    """What the runs together say: stable pass, flaky, stable fail or error.

    An error is only the label when nothing plainly failed. A test that failed
    once and errored once is flaky or a stable fail, and the error is still
    counted on its own.
    """
    if not runs:
        return ERROR
    passes = [one for one in runs if one.status == STATUS_PASS]
    errors = [one for one in runs if one.status == STATUS_ERROR]
    failures = [one for one in runs if one.status not in (STATUS_PASS, STATUS_ERROR)]

    if errors and not failures:
        return ERROR
    if len(passes) == len(runs):
        return STABLE_PASS
    if passes:
        return FLAKY
    return STABLE_FAIL


def _added_tokens(runs: list[TestOutcome], field_name: str) -> TokenUsage:
    total = TokenUsage()
    for one in runs:
        usage = getattr(one, field_name, None)
        if usage is not None:
            total = total.plus(usage)
    return total


@dataclass
class RepeatedTest:
    """One test case and every run of it, in the order they ran."""

    test_case: TestCase
    runs: list[TestOutcome]

    # ---------- what the runs say together ----------

    @property
    def stability(self) -> str:
        return stability_of(self.runs)

    @property
    def runs_total(self) -> int:
        return len(self.runs)

    @property
    def runs_passed(self) -> int:
        return len([one for one in self.runs if one.status == STATUS_PASS])

    @property
    def runs_text(self) -> str:
        """2/3, for the terminal line and the report."""
        return str(self.runs_passed) + "/" + str(self.runs_total)

    @property
    def is_flaky(self) -> bool:
        return self.stability == FLAKY

    # ---------- the same questions a single outcome answers ----------

    @property
    def status(self) -> str:
        return self.stability

    @property
    def passed(self) -> bool:
        """Only when every run passed. One bad run is enough to fail the test."""
        return self.stability == STABLE_PASS

    @property
    def is_error(self) -> bool:
        return self.stability == ERROR

    @property
    def is_critical_failure(self) -> bool:
        """A critical failure in any run counts for the whole test."""
        return any(one.is_critical_failure for one in self.runs)

    @property
    def agent_tokens(self) -> TokenUsage:
        return _added_tokens(self.runs, "agent_tokens")

    @property
    def judge_tokens(self) -> TokenUsage:
        return _added_tokens(self.runs, "judge_tokens")

    @property
    def duration_ms(self) -> float:
        return round(sum(float(getattr(one, "duration_ms", 0.0)) for one in self.runs), 3)

    def area_passed(self) -> dict[str, bool]:
        """An area passed only when it passed in every run."""
        if not self.runs:
            return {area: False for area in ALL_AREAS}
        per_run = [one.area_passed() for one in self.runs]
        return {area: all(one.get(area, False) for one in per_run) for area in ALL_AREAS}

    # ---------- the runs a person wants to read ----------

    @property
    def first_failed_run(self) -> TestOutcome | None:
        """The earliest run that did not pass. What the terminal line shows."""
        for one in self.runs:
            if one.status != STATUS_PASS:
                return one
        return None

    @property
    def first_failed_run_number(self) -> int | None:
        """Run numbers start at 1, the way the terminal and the report count."""
        for number, one in enumerate(self.runs, start=1):
            if one.status != STATUS_PASS:
                return number
        return None

    @property
    def main_message(self) -> str:
        """The one line that says what happened. Empty when every run passed."""
        worst = self.first_failed_run
        return worst.first_failure_message if worst is not None else ""

    @property
    def main_area(self) -> str | None:
        """The area of the worst failed check in the first failed run."""
        worst = self.first_failed_run
        if worst is None:
            return None
        if worst.is_error:
            return None
        check = worst.worst_failed_check
        return check.area if check is not None else None

    def to_dict(self) -> dict[str, Any]:
        """Everything a single test held before, plus the runs behind it."""
        newest = self.runs[-1] if self.runs else None
        as_dict = dict(newest.to_dict()) if newest is not None else {"id": self.test_case.id}
        as_dict.update(
            {
                "status": self.stability if self.runs_total > 1 else as_dict.get("status"),
                "passed": self.passed,
                "critical_failure": self.is_critical_failure,
                "stability": self.stability,
                "runs_passed": self.runs_passed,
                "runs_total": self.runs_total,
                "areas": self.area_passed(),
                "agent_tokens": self.agent_tokens.to_dict(),
                "judge_tokens": self.judge_tokens.to_dict(),
                "duration_ms": self.duration_ms,
                "runs": [
                    _one_run_dict(one, number, self.test_case.id, self.runs_total)
                    for number, one in enumerate(self.runs, start=1)
                ],
            }
        )
        as_dict["trace_file"] = trace_path_for(self.test_case.id, 1, self.runs_total)
        return as_dict


def _one_run_dict(outcome: TestOutcome, number: int, test_id: str, runs_total: int) -> dict[str, Any]:
    """One run inside a test: what a test held before, plus its run number."""
    as_dict = dict(outcome.to_dict())
    as_dict["run"] = number
    as_dict["trace_file"] = trace_path_for(test_id, number, runs_total)
    return as_dict


def trace_path_for(test_id: str, run_number: int, runs_total: int) -> str:
    """Where one run's trace is saved, relative to the run folder.

    With one run the old layout is kept, so older tools and older reports still
    find the file. With more than one, each test gets a folder of its own.
    """
    if runs_total <= 1:
        return "traces/" + test_id + ".json"
    return "traces/" + test_id + "/run-" + str(run_number) + ".json"
