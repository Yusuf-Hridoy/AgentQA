"""Did the agent take the right steps?

Right tools, right order, not too many calls, and no repeated work.
A call counts here even if it was blocked or failed, because the agent still made it.
"""

from __future__ import annotations

import json

from agentqa.checks.check_result import AREA_STEPS, CheckResult, failed, passed
from agentqa.test_case_schema import TestCase
from agentqa.trace import Trace


def check_should_use_in_order(test_case: TestCase, trace: Trace) -> CheckResult | None:
    """The listed tools must be used in this order. Other calls may come between them."""
    wanted = list(test_case.should_use)
    if not wanted:
        return None

    used = trace.called_tool_names_in_order()
    position = 0
    for name in used:
        if position < len(wanted) and name == wanted[position]:
            position += 1

    name_of_check = "right_steps"
    if position == len(wanted):
        return passed(name_of_check, "Used the right steps: " + " then ".join(wanted) + ".", AREA_STEPS)

    missing = wanted[position]
    if missing not in used:
        return failed(
            name_of_check,
            "Never used " + missing + ". Expected: " + " then ".join(wanted)
            + ". Used: " + (", ".join(used) if used else "no tools") + ".",
            AREA_STEPS,
        )
    return failed(
        name_of_check,
        "Wrong order. Expected: " + " then ".join(wanted)
        + ". Used: " + ", ".join(used) + ".",
        AREA_STEPS,
    )


def check_max_calls(test_case: TestCase, trace: Trace) -> list[CheckResult]:
    """A tool may not be called more often than the test case allows."""
    results: list[CheckResult] = []
    for tool, limit in test_case.max_calls.items():
        used = trace.count_calls(tool)
        name_of_check = "max_calls_" + tool
        if used > limit:
            results.append(
                failed(
                    name_of_check,
                    "Called " + tool + " " + str(used) + " times. Expected at most " + str(limit) + ".",
                    AREA_STEPS,
                )
            )
        else:
            results.append(
                passed(
                    name_of_check,
                    "Called " + tool + " " + str(used) + " times. The limit is " + str(limit) + ".",
                    AREA_STEPS,
                )
            )
    return results


def check_no_repeated_calls(test_case: TestCase, trace: Trace) -> CheckResult:
    """The same tool with the same input must not succeed twice. That is double work."""
    seen: set[str] = set()
    repeats: list[str] = []
    for event in trace.successful_calls():
        key = str(event.tool) + " " + json.dumps(event.inputs or {}, sort_keys=True, default=str)
        if key in seen and str(event.tool) not in repeats:
            repeats.append(str(event.tool))
        seen.add(key)

    name_of_check = "no_repeated_calls"
    if repeats:
        return failed(
            name_of_check,
            "Did the same call twice: " + ", ".join(repeats) + ".",
            AREA_STEPS,
        )
    return passed(name_of_check, "No call was made twice with the same input.", AREA_STEPS)


def run_tool_call_checks(test_case: TestCase, trace: Trace) -> list[CheckResult]:
    results: list[CheckResult] = []
    order_result = check_should_use_in_order(test_case, trace)
    if order_result is not None:
        results.append(order_result)
    results.extend(check_max_calls(test_case, trace))
    results.append(check_no_repeated_calls(test_case, trace))
    return results
