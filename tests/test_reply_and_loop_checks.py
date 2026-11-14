"""Reply checks and loop checks."""

from __future__ import annotations

from conftest import failed_call, make_test_case, make_trace, ok_call

from agentqa.checks.loop_checks import check_retry_limit, check_step_limit
from agentqa.checks.reply_checks import (
    check_expect_question,
    check_reply_exists,
    check_reply_should_say,
)


def test_reply_should_say_passes_in_any_case():
    case = make_test_case(reply_should_say=["ticket", "VPN"])

    result = check_reply_should_say(case, "I made a Ticket for your vpn problem.")

    assert result is not None and result.passed
    assert result.area == "task"


def test_reply_should_say_names_the_missing_words():
    case = make_test_case(reply_should_say=["ticket", "vpn"])

    result = check_reply_should_say(case, "I made a ticket.")

    assert not result.passed
    assert "vpn" in result.message


def test_reply_should_say_fails_when_there_is_no_reply():
    case = make_test_case(reply_should_say=["ticket"])

    result = check_reply_should_say(case, None)

    assert not result.passed
    assert "no reply" in result.message


def test_expect_question_passes_on_a_question():
    case = make_test_case(expect_question=True)

    assert check_expect_question(case, "Which device is broken?").passed


def test_expect_question_fails_without_one():
    case = make_test_case(expect_question=True)

    result = check_expect_question(case, "I made a ticket for you.")

    assert not result.passed
    assert "asks no question" in result.message


def test_expect_question_is_skipped_when_not_asked_for():
    assert check_expect_question(make_test_case(), "No question here.") is None


def test_reply_exists_passes_and_fails():
    assert check_reply_exists("Here you go.").passed
    assert not check_reply_exists(None).passed
    assert not check_reply_exists("   ").passed


def test_step_limit_passes_when_the_agent_answered():
    case = make_test_case(max_steps=6)
    trace = make_trace([ok_call("get_ticket")], reply="Done.")

    result = check_step_limit(case, trace)

    assert result.passed
    assert result.area == "steps"


def test_step_limit_fails_when_the_agent_ran_out_of_steps():
    case = make_test_case(max_steps=3)
    trace = make_trace([ok_call("get_ticket")], hit_step_limit=True)

    result = check_step_limit(case, trace)

    assert not result.passed
    assert "never answered" in result.message


def test_retry_limit_passes_within_the_allowed_tries():
    case = make_test_case(tool_failures={"create_ticket": 5}, max_calls={"create_ticket": 3})
    trace = make_trace([failed_call("create_ticket")] * 3, reply="I could not make it.")

    results = check_retry_limit(case, trace)

    assert len(results) == 1 and results[0].passed


def test_retry_limit_fails_when_the_agent_keeps_trying():
    case = make_test_case(tool_failures={"create_ticket": 9}, max_calls={"create_ticket": 3})
    trace = make_trace([failed_call("create_ticket")] * 5, reply="I could not make it.")

    result = check_retry_limit(case, trace)[0]

    assert not result.passed
    assert "5 times" in result.message


def test_retry_limit_uses_a_default_when_the_test_sets_none():
    case = make_test_case(tool_failures={"create_ticket": 9})
    trace = make_trace([failed_call("create_ticket")] * 4, reply="I could not make it.")

    assert not check_retry_limit(case, trace)[0].passed
