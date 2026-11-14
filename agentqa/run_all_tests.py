"""Runs each test case and collects the results.

One test = one fresh ticket store, one brain, one agent loop, then the checks.
Nothing is shared between tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentqa.agent_loop import run_agent_loop
from agentqa.brains.brain_factory import build_brain
from agentqa.brains.brain_interface import Brain
from agentqa.checks.check_result import ALL_AREAS, CheckResult
from agentqa.checks.loop_checks import run_loop_checks
from agentqa.checks.reply_checks import run_reply_checks
from agentqa.checks.safety_checks import run_safety_checks
from agentqa.checks.ticket_system_checks import run_ticket_system_checks
from agentqa.checks.tool_call_checks import run_tool_call_checks
from agentqa.fake_ticket_system.ticket_store import TicketStore
from agentqa.fake_ticket_system.tool_gateway import ToolGateway
from agentqa.test_case_schema import TestCase
from agentqa.trace import Trace


@dataclass
class TestOutcome:
    """Everything one test produced."""

    test_case: TestCase
    checks: list[CheckResult]
    trace: Trace
    final_reply: str | None
    steps_used: int
    ticket_system: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [check for check in self.checks if not check.passed]

    @property
    def first_failure_message(self) -> str:
        failures = self.failed_checks
        return failures[0].message if failures else ""

    @property
    def is_critical_failure(self) -> bool:
        return self.test_case.critical and not self.passed

    def area_passed(self) -> dict[str, bool]:
        """Per area: did every check in that area pass?

        A failure in a critical test also counts against safety.
        """
        result = {}
        for area in ALL_AREAS:
            in_area = [check for check in self.checks if check.area == area]
            result[area] = all(check.passed for check in in_area)
        if self.is_critical_failure:
            result["safety"] = False
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.test_case.id,
            "type": self.test_case.type,
            "acting_user": self.test_case.acting_user,
            "user_asks": self.test_case.user_asks,
            "critical": self.test_case.critical,
            "source_file": self.test_case.source_file,
            "passed": self.passed,
            "critical_failure": self.is_critical_failure,
            "duration_ms": self.duration_ms,
            "steps_used": self.steps_used,
            "final_reply": self.final_reply,
            "areas": self.area_passed(),
            "checks": [check.model_dump() for check in self.checks],
            "failed_checks": [check.model_dump() for check in self.failed_checks],
            "ticket_system": self.ticket_system,
            "trace_file": "traces/" + self.test_case.id + ".json",
        }


def run_checks(test_case: TestCase, trace: Trace, store: TicketStore, final_reply: str | None) -> list[CheckResult]:
    """Every check, in a fixed order. Nothing is averaged away."""
    results: list[CheckResult] = []
    results.extend(run_tool_call_checks(test_case, trace))
    results.extend(run_loop_checks(test_case, trace))
    results.extend(run_safety_checks(test_case, trace, final_reply))
    results.extend(run_ticket_system_checks(test_case, store))
    results.extend(run_reply_checks(test_case, final_reply))
    return results


def run_one_test(
    test_case: TestCase,
    brain_name: str = "fake",
    brain: Brain | None = None,
    scripts_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> TestOutcome:
    """Run one test case from a fresh copy of the fake data."""
    started = time.perf_counter()

    store = TicketStore.load_fresh(data_dir)
    trace = Trace(test_id=test_case.id, acting_user=test_case.acting_user)
    gateway = ToolGateway(
        store=store,
        acting_user=test_case.acting_user,
        trace=trace,
        tool_failures=test_case.tool_failures,
    )
    thinking = brain if brain is not None else build_brain(brain_name, test_case.id, scripts_dir)

    run_result = run_agent_loop(brain=thinking, gateway=gateway, test_case=test_case)
    checks = run_checks(test_case, trace, store, run_result.final_reply)

    return TestOutcome(
        test_case=test_case,
        checks=checks,
        trace=trace,
        final_reply=run_result.final_reply,
        steps_used=run_result.steps_used,
        ticket_system=store.snapshot(),
        duration_ms=round((time.perf_counter() - started) * 1000, 3),
    )


def run_all_tests(
    test_cases: list[TestCase],
    brain_name: str = "fake",
    scripts_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    on_result: Any = None,
) -> list[TestOutcome]:
    """Run every test case. on_result is called after each test, for live output."""
    outcomes: list[TestOutcome] = []
    for test_case in test_cases:
        outcome = run_one_test(
            test_case,
            brain_name=brain_name,
            scripts_dir=scripts_dir,
            data_dir=data_dir,
        )
        outcomes.append(outcome)
        if on_result is not None:
            on_result(outcome)
    return outcomes
