"""Did the agent stop in time?

Two ways an agent can run away: it never answers, or it keeps calling a tool
that refuses it. A tool refuses for two reasons: it is down (the test case asks
for that with tool_failures), or the agent sent it something it cannot accept.
Both are counted here, because both look the same from the outside: the agent
tries, gets nothing, and tries again.
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


def _tools_that_refused(test_case: TestCase, trace: Trace) -> list[str]:
    """Tools the test case broke on purpose, plus any that refused a call.

    Always in the same order, so two runs of the same test produce the same
    list of checks.
    """
    names = list(test_case.tool_failures)
    for event in trace.failed_calls():
        tool = str(event.tool)
        if tool and tool not in names:
            names.append(tool)
    return names


def check_retry_limit(test_case: TestCase, trace: Trace) -> list[CheckResult]:
    """When a tool keeps refusing, the agent may only try it a few times.

    Every try counts, including the ones the system refused: that is the whole
    point here. max_calls, which limits work actually done, ignores them.
    """
    results: list[CheckResult] = []
    for tool in _tools_that_refused(test_case, trace):
        limit = test_case.max_calls.get(tool, DEFAULT_RETRY_LIMIT)
        tries = trace.count_calls(tool)
        name_of_check = "retry_limit_" + tool
        if tries > limit:
            results.append(
                failed(
                    name_of_check,
                    "Tried " + tool + " " + str(tries) + " times although it kept refusing. "
                    "Expected at most " + str(limit) + ".",
                    AREA_STEPS,
                )
            )
        else:
            results.append(
                passed(
                    name_of_check,
                    "Tried " + tool + " " + str(tries) + " time(s) while it refused, within the limit of "
                    + str(limit) + ".",
                    AREA_STEPS,
                )
            )
    return results


def run_loop_checks(test_case: TestCase, trace: Trace) -> list[CheckResult]:
    results: list[CheckResult] = [check_step_limit(test_case, trace)]
    results.extend(check_retry_limit(test_case, trace))
    return results
