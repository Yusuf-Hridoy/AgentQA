"""Safety checks: forbidden tools, blocked calls and leaking replies."""

from __future__ import annotations

from conftest import blocked_call, make_test_case, make_trace, ok_call

from agentqa.checks.safety_checks import (
    check_blocked_calls,
    check_must_never_use,
    check_reply_must_not_say,
)


def test_forbidden_tools_passes_when_they_are_left_alone():
    case = make_test_case(must_never_use=["assign_ticket", "send_notification"])
    trace = make_trace([ok_call("get_ticket")], reply="Done.")

    result = check_must_never_use(case, trace)

    assert result is not None and result.passed
    assert result.area == "safety"


def test_forbidden_tools_fails_when_one_is_used():
    case = make_test_case(must_never_use=["send_notification"])
    trace = make_trace([ok_call("send_notification", {"to": "all-staff", "message": "hi"})], reply="Sent.")

    result = check_must_never_use(case, trace)

    assert not result.passed
    assert "send_notification" in result.message


def test_forbidden_tools_fails_even_when_the_gateway_blocked_the_call():
    case = make_test_case(must_never_use=["assign_ticket"])
    trace = make_trace([blocked_call("assign_ticket", {"ticket_id": "TCK-1001", "assignee": "bob"})], reply="No.")

    result = check_must_never_use(case, trace)

    assert not result.passed


def test_no_forbidden_list_means_no_check():
    assert check_must_never_use(make_test_case(), make_trace(reply="Done.")) is None


def test_blocked_calls_pass_when_nothing_was_blocked():
    result = check_blocked_calls(make_test_case(), make_trace([ok_call("get_ticket")], reply="Done."))

    assert result.passed


def test_blocked_calls_fail_when_the_test_did_not_expect_them():
    case = make_test_case()
    trace = make_trace([blocked_call("update_ticket")], reply="Sorry.")

    result = check_blocked_calls(case, trace)

    assert not result.passed
    assert "update_ticket" in result.message


def test_blocked_calls_pass_when_the_test_expects_them():
    case = make_test_case(expect_blocked=["get_ticket"])
    trace = make_trace([blocked_call("get_ticket")], reply="Sorry.")

    result = check_blocked_calls(case, trace)

    assert result.passed
    assert "expects" in result.message


def test_reply_must_not_say_passes_when_the_word_is_absent():
    case = make_test_case(reply_must_not_say=["payroll"])

    result = check_reply_must_not_say(case, "I am not allowed to open that ticket.")

    assert result is not None and result.passed


def test_reply_must_not_say_fails_on_any_upper_or_lower_case():
    case = make_test_case(reply_must_not_say=["payroll"])

    result = check_reply_must_not_say(case, "The ticket is a Payroll access request.")

    assert not result.passed
    assert "payroll" in result.message


def test_reply_must_not_say_is_skipped_when_the_list_is_empty():
    assert check_reply_must_not_say(make_test_case(), "anything") is None
