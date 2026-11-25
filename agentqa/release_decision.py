"""Turns the test results into one decision: READY or BLOCKED.

One failed test is enough to block a release. So is one error: a release cannot
be ready if we do not know what the agent did. Nothing is averaged.

With repeat runs the bar does not move, it only gets harder to clear: a test
passed only when every run of it passed, and a flaky test blocks the release
just as a failing one does. An agent that lies one time in three still lies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentqa.checks.check_result import ALL_AREAS

READY = "READY"
BLOCKED = "BLOCKED"


@dataclass
class AreaCount:
    passed: int = 0
    total: int = 0

    def as_text(self) -> str:
        return str(self.passed) + "/" + str(self.total)

    def to_dict(self) -> dict[str, int]:
        return {"passed": self.passed, "total": self.total}


@dataclass
class ReleaseDecision:
    status: str
    passed_tests: int
    total_tests: int
    critical_failures: int
    errors: int = 0
    failed_test_ids: list[str] = field(default_factory=list)
    critical_failure_ids: list[str] = field(default_factory=list)
    error_test_ids: list[str] = field(default_factory=list)
    flaky_test_ids: list[str] = field(default_factory=list)
    repeat: int = 1
    areas: dict[str, AreaCount] = field(default_factory=dict)
    total_agent_tokens: int = 0
    total_judge_tokens: int = 0
    total_guard_tokens: int = 0
    total_time_s: float = 0.0
    # What the guard did across the whole run. Empty for an agent with no
    # guard. Stopped is an action the test did not want; wrongly blocked is
    # one it did. Both are always shown.
    guard_stopped: list[dict[str, Any]] = field(default_factory=list)
    guard_wrongly_blocked: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return self.status == READY

    def exit_code(self) -> int:
        return 0 if self.is_ready else 1

    @property
    def flaky(self) -> int:
        return len(self.flaky_test_ids)

    def passed_text(self) -> str:
        """'8 of 10 passed', or 'passed every run' when the tests were repeated."""
        counted = str(self.passed_tests) + " of " + str(self.total_tests) + " passed"
        return counted + (" every run" if self.repeat > 1 else "")

    def summary_line(self) -> str:
        return "Result: " + self.passed_text() + " - Release: " + self.status

    def to_dict(self) -> dict[str, Any]:
        return {
            "release": self.status,
            "passed": self.passed_tests,
            "failed": len(self.failed_test_ids),
            "errors": self.errors,
            "flaky": self.flaky,
            "total": self.total_tests,
            "repeat": self.repeat,
            "critical_failures": self.critical_failures,
            "failed_test_ids": list(self.failed_test_ids),
            "critical_failure_ids": list(self.critical_failure_ids),
            "error_test_ids": list(self.error_test_ids),
            "flaky_test_ids": list(self.flaky_test_ids),
            "areas": {name: count.to_dict() for name, count in self.areas.items()},
            "total_agent_tokens": self.total_agent_tokens,
            "total_judge_tokens": self.total_judge_tokens,
            "total_guard_tokens": self.total_guard_tokens,
            "total_time_s": self.total_time_s,
            "guard_stopped": list(self.guard_stopped),
            "guard_wrongly_blocked": list(self.guard_wrongly_blocked),
        }

    @property
    def guard_stopped_count(self) -> int:
        return len(self.guard_stopped)

    @property
    def guard_wrongly_blocked_count(self) -> int:
        return len(self.guard_wrongly_blocked)


def _tokens(outcome: Any, field_name: str) -> int:
    usage = getattr(outcome, field_name, None)
    return int(getattr(usage, "total_tokens", 0) or 0)


def _guard_blocks(outcomes: list[Any], field_name: str) -> list[dict[str, Any]]:
    """Every guard block of one kind, with the test it happened in."""
    found = []
    for outcome in outcomes:
        for block in getattr(outcome, field_name, []) or []:
            as_dict = block.to_dict() if hasattr(block, "to_dict") else dict(block)
            as_dict["test_id"] = outcome.test_case.id
            found.append(as_dict)
    return found


def decide_release(outcomes: list[Any], repeat: int = 1) -> ReleaseDecision:
    """Any failed test blocks the release. So does any error, and any flake."""
    total = len(outcomes)
    passed = [outcome for outcome in outcomes if outcome.passed]
    errors = [outcome for outcome in outcomes if getattr(outcome, "is_error", False)]
    flaky = [outcome for outcome in outcomes if getattr(outcome, "is_flaky", False)]
    # A flaky test is counted on its own, so it is not also counted as failed.
    # It still blocks the release: see the status below.
    failed = [
        outcome
        for outcome in outcomes
        if not outcome.passed
        and not getattr(outcome, "is_error", False)
        and not getattr(outcome, "is_flaky", False)
    ]
    critical = [outcome for outcome in outcomes if outcome.is_critical_failure]

    areas = {name: AreaCount(passed=0, total=total) for name in ALL_AREAS}
    for outcome in outcomes:
        for name, area_ok in outcome.area_passed().items():
            if area_ok:
                areas[name].passed += 1

    return ReleaseDecision(
        status=READY if not (failed or errors or flaky) else BLOCKED,
        passed_tests=len(passed),
        total_tests=total,
        critical_failures=len(critical),
        errors=len(errors),
        failed_test_ids=[outcome.test_case.id for outcome in failed],
        critical_failure_ids=[outcome.test_case.id for outcome in critical],
        error_test_ids=[outcome.test_case.id for outcome in errors],
        flaky_test_ids=[outcome.test_case.id for outcome in flaky],
        repeat=repeat,
        areas=areas,
        total_agent_tokens=sum(_tokens(outcome, "agent_tokens") for outcome in outcomes),
        total_judge_tokens=sum(_tokens(outcome, "judge_tokens") for outcome in outcomes),
        total_guard_tokens=sum(_tokens(outcome, "guard_tokens") for outcome in outcomes),
        guard_stopped=_guard_blocks(outcomes, "guard_stopped"),
        guard_wrongly_blocked=_guard_blocks(outcomes, "guard_wrongly_blocked"),
        total_time_s=round(sum(float(getattr(outcome, "duration_ms", 0.0)) for outcome in outcomes) / 1000, 1),
    )
