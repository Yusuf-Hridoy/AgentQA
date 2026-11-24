"""Runs the bug zoo: every broken agent must be caught by the right check.

The zoo is how we test AgentQA itself. Eight agents are broken on purpose, each
in one known way. If a bug walks through, AgentQA has a blind spot, and the zoo
says so instead of showing a green run.

Everything here uses fake brains and the word claim reader, so the zoo needs no
key and no network and runs in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentqa.brains.fake_brain import FakeBrain
from agentqa.claims.word_claim_reader import WordClaimReader
from agentqa.project_paths import TEST_CASES_DIR
from agentqa.run_all_tests import run_one_test
from agentqa.test_case_loader import load_test_cases
from agentqa.zoo.zoo_loader import BrokenAgent, ZooFileError, load_broken_agents

CAUGHT = "CAUGHT"
CAUGHT_BY_THE_WRONG_CHECK = "CAUGHT (wrong check)"
MISSED = "MISSED"


@dataclass
class ZooResult:
    """What happened to one broken agent."""

    agent: BrokenAgent
    result: str
    failed_checks: list[str] = field(default_factory=list)
    catching_message: str = ""
    test_status: str = ""

    @property
    def is_caught(self) -> bool:
        return self.result == CAUGHT

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.agent.name,
            "bug": self.agent.bug,
            "based_on": self.agent.based_on,
            "test_case": self.agent.test_case,
            "expected_check": self.agent.expected_check,
            "result": self.result,
            "test_status": self.test_status,
            "failed_checks": list(self.failed_checks),
            "catching_message": self.catching_message,
        }


def run_broken_agent(
    agent: BrokenAgent,
    cases_folder: str | Path | None = None,
) -> ZooResult:
    """Run one broken agent against its test case and see what the checks say."""
    folder = Path(cases_folder) if cases_folder else TEST_CASES_DIR
    found = load_test_cases(folder, only=agent.test_case)
    test_case = found[0]

    outcome = run_one_test(
        test_case,
        brain=FakeBrain.from_script_file(agent.script_file),
        claim_reader=WordClaimReader(),
    )

    failed = [check.check_name for check in outcome.failed_checks]
    if outcome.passed:
        return ZooResult(agent=agent, result=MISSED, test_status=outcome.status)

    if agent.expected_check in failed:
        message = next(
            check.message
            for check in outcome.failed_checks
            if check.check_name == agent.expected_check
        )
        return ZooResult(
            agent=agent,
            result=CAUGHT,
            failed_checks=failed,
            catching_message=message,
            test_status=outcome.status,
        )

    return ZooResult(
        agent=agent,
        result=CAUGHT_BY_THE_WRONG_CHECK,
        failed_checks=failed,
        catching_message=outcome.first_failure_message,
        test_status=outcome.status,
    )


def run_zoo(
    zoo_folder: str | Path | None = None,
    cases_folder: str | Path | None = None,
    only: str | None = None,
    on_result: Any = None,
) -> list[ZooResult]:
    """Run every broken agent. on_result is called after each one, for live output."""
    results = []
    for agent in load_broken_agents(zoo_folder, only=only):
        result = run_broken_agent(agent, cases_folder=cases_folder)
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results


def all_caught(results: list[ZooResult]) -> bool:
    return bool(results) and all(result.is_caught for result in results)


def zoo_exit_code(results: list[ZooResult]) -> int:
    """0 only when every bug was caught by the check that should catch it."""
    return 0 if all_caught(results) else 1


__all__ = [
    "CAUGHT",
    "CAUGHT_BY_THE_WRONG_CHECK",
    "MISSED",
    "ZooFileError",
    "ZooResult",
    "all_caught",
    "run_broken_agent",
    "run_zoo",
    "zoo_exit_code",
]
