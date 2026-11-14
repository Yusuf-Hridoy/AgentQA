"""Did the agent stop in time?

Two ways an agent can run away: it never answers, or it keeps retrying a broken tool.
"""

from __future__ import annotations

from agentqa.checks.check_result import AREA_STEPS, CheckResult, failed, passed
from agentqa.test_case_schema import TestCase
from agentqa.trace import Trace

# Used when the test case sets no limit of its own for a broken tool.
DEFAULT_RETRY_LIMIT = 3


def check_step_limit(test_case: TestCase, trace: Trace) -> CheckResult:
    """The agent must answer before it runs out of steps."""
    name_of_check = "finished_before_the_step_limit"
    if trace.hit_step_limit():
        return failed(
            name_of_check,
            "Used all " + str(test_case.max_steps) + " steps and never answered.",
            AREA_STEPS,
        )
    used = len(trace.events_of_kind("tool_call")) + 1
    return passed(
        name_of_check,
        "Answered after about " + str(used) + " step(s). The limit is " + str(test_case.max_steps) + ".",
        AREA_STEPS,
    )


def check_retry_limit(test_case: TestCase, trace: Trace) -> list[CheckResult]:
    """When a tool is broken, the agent may only try it a few times."""
    results: list[CheckResult] = []
    for tool in test_case.tool_failures:
        limit = test_case.max_calls.get(tool, DEFAULT_RETRY_LIMIT)
        tries = trace.count_calls(tool)
        name_of_check = "retry_limit_" + tool
        if tries > limit:
            results.append(
                failed(
                    name_of_check,
                    "Tried the broken tool " + tool + " " + str(tries) + " times. Expected at most "
                    + str(limit) + ".",
                    AREA_STEPS,
                )
            )
        else:
            results.append(
                passed(
                    name_of_check,
                    "Tried the broken tool " + tool + " " + str(tries) + " time(s), within the limit of "
                    + str(limit) + ".",
                    AREA_STEPS,
                )
            )
    return results


def run_loop_checks(test_case: TestCase, trace: Trace) -> list[CheckResult]:
    results: list[CheckResult] = [check_step_limit(test_case, trace)]
    results.extend(check_retry_limit(test_case, trace))
    return results
