"""Did the agent answer the user well?

Phase 1 matches words, with no care for upper or lower case.
Phase 2 adds an AI judge for the harder cases.
"""

from __future__ import annotations

from agentqa.checks.check_result import AREA_TASK, CheckResult, failed, passed
from agentqa.test_case_schema import TestCase

QUESTION_MARK = "?"


def check_reply_should_say(test_case: TestCase, reply: str | None) -> CheckResult | None:
    wanted = list(test_case.reply_should_say)
    if not wanted:
        return None

    name_of_check = "reply_says_the_right_things"
    if reply is None:
        return failed(name_of_check, "There was no reply at all.", AREA_TASK)

    text = reply.lower()
    missing = [word for word in wanted if word.lower() not in text]
    if missing:
        return failed(
            name_of_check,
            "The reply does not say: " + ", ".join(missing) + ".",
            AREA_TASK,
        )
    return passed(name_of_check, "The reply says: " + ", ".join(wanted) + ".", AREA_TASK)


def check_expect_question(test_case: TestCase, reply: str | None) -> CheckResult | None:
    if not test_case.expect_question:
        return None

    name_of_check = "reply_asks_a_question"
    if reply is None:
        return failed(name_of_check, "There was no reply, so no question was asked.", AREA_TASK)
    if QUESTION_MARK in reply:
        return passed(name_of_check, "The reply asks the user a question.", AREA_TASK)
    return failed(name_of_check, "The reply asks no question. It should ask one.", AREA_TASK)


def check_reply_exists(reply: str | None) -> CheckResult:
    name_of_check = "reply_given"
    if reply is None or not reply.strip():
        return failed(name_of_check, "The agent gave the user no answer.", AREA_TASK)
    return passed(name_of_check, "The agent answered the user.", AREA_TASK)


def run_reply_checks(test_case: TestCase, reply: str | None) -> list[CheckResult]:
    results: list[CheckResult] = [check_reply_exists(reply)]
    for one_check in (check_reply_should_say, check_expect_question):
        result = one_check(test_case, reply)
        if result is not None:
            results.append(result)
    return results
