"""Turns the test results into one decision: READY or BLOCKED.

One failed test is enough to block a release. Nothing is averaged.
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
    failed_test_ids: list[str] = field(default_factory=list)
    critical_failure_ids: list[str] = field(default_factory=list)
    areas: dict[str, AreaCount] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        return self.status == READY

    def exit_code(self) -> int:
        return 0 if self.is_ready else 1

    def summary_line(self) -> str:
        return (
            "Result: " + str(self.passed_tests) + " of " + str(self.total_tests)
            + " passed - Release: " + self.status
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "release": self.status,
            "passed": self.passed_tests,
            "failed": self.total_tests - self.passed_tests,
            "total": self.total_tests,
            "critical_failures": self.critical_failures,
            "failed_test_ids": list(self.failed_test_ids),
            "critical_failure_ids": list(self.critical_failure_ids),
            "areas": {name: count.to_dict() for name, count in self.areas.items()},
        }


def decide_release(outcomes: list[Any]) -> ReleaseDecision:
    """Any failed test blocks the release."""
    total = len(outcomes)
    passed = [outcome for outcome in outcomes if outcome.passed]
    failed = [outcome for outcome in outcomes if not outcome.passed]
    critical = [outcome for outcome in outcomes if outcome.is_critical_failure]

    areas = {name: AreaCount(passed=0, total=total) for name in ALL_AREAS}
    for outcome in outcomes:
        for name, area_ok in outcome.area_passed().items():
            if area_ok:
                areas[name].passed += 1

    return ReleaseDecision(
        status=READY if not failed else BLOCKED,
        passed_tests=len(passed),
        total_tests=total,
        critical_failures=len(critical),
        failed_test_ids=[outcome.test_case.id for outcome in failed],
        critical_failure_ids=[outcome.test_case.id for outcome in critical],
        areas=areas,
    )
