"""Did the agent answer the user well?

Two ways to check a reply:

words   reply_should_say, matched with no care for upper or lower case.
        Fast, free and exact. Right for the fake brain and for CI.
judge   reply_should_mean, read by the AI judge.
        Right for real models, which say the same thing in other words.

reply_must_not_say and expect_question stay plain code either way. A leaked
word is a fact, not an opinion.
"""

from __future__ import annotations

from agentqa.checks.check_result import AREA_TASK, CheckResult, failed, judged, passed
from agentqa.test_case_schema import TestCase

QUESTION_MARK = "?"
JUDGE_CHECK_NAME = "reply_means_the_right_thing"


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
        return failed(name_of_check, "The agent gave no reply.", AREA_TASK)
    return passed(name_of_check, "The agent answered the user.", AREA_TASK)


def check_reply_should_mean(test_case: TestCase, reply: str | None, judge: object) -> CheckResult | None:
    """Ask the judge what the reply means. A judge failure raises, it never passes."""
    expectation = (test_case.reply_should_mean or "").strip()
    if judge is None or not expectation:
        return None

    if reply is None or not reply.strip():
        return failed(JUDGE_CHECK_NAME, "There was no reply to judge.", AREA_TASK)

    verdict = judge.judge(test_case.user_asks, reply, expectation)
    return judged(JUDGE_CHECK_NAME, verdict.passed, verdict.reason, AREA_TASK)


def run_reply_checks(test_case: TestCase, reply: str | None, judge: object = None) -> list[CheckResult]:
    """The judge replaces the word check, it never runs beside it."""
    reply_exists = check_reply_exists(reply)
    if not reply_exists.passed:
        # There is nothing to match, to mean or to ask. Stopping here also
        # keeps an empty reply away from the judge, which must never grade
        # text the agent did not write.
        return [reply_exists]

    results: list[CheckResult] = [reply_exists]

    judged_result = check_reply_should_mean(test_case, reply, judge)
    if judged_result is not None:
        results.append(judged_result)
    else:
        word_result = check_reply_should_say(test_case, reply)
        if word_result is not None:
            results.append(word_result)

    question_result = check_expect_question(test_case, reply)
    if question_result is not None:
        results.append(question_result)

    return results
