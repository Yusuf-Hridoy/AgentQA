"""Steps checks: right tools, right order, call limits, no repeated work."""

from __future__ import annotations

from conftest import blocked_call, failed_call, make_test_case, make_trace, ok_call

from agentqa.checks.tool_call_checks import (
    check_max_calls,
    check_no_repeated_calls,
    check_should_use_in_order,
)


def test_right_steps_passes_when_the_order_is_kept():
    case = make_test_case(should_use=["search_help_docs", "create_ticket"])
    trace = make_trace([ok_call("search_help_docs"), ok_call("create_ticket")], reply="Done.")

    result = check_should_use_in_order(case, trace)

    assert result is not None and result.passed
    assert result.area == "steps"


def test_right_steps_allows_other_calls_in_between():
    case = make_test_case(should_use=["search_help_docs", "create_ticket"])
    trace = make_trace(
        [ok_call("search_help_docs"), ok_call("get_ticket"), ok_call("create_ticket")],
        reply="Done.",
    )

    assert check_should_use_in_order(case, trace).passed


def test_right_steps_fails_when_a_step_is_missing():
    case = make_test_case(should_use=["search_help_docs", "create_ticket"])
    trace = make_trace([ok_call("search_help_docs")], reply="Done.")

    result = check_should_use_in_order(case, trace)

    assert not result.passed
    assert "Never used create_ticket" in result.message


def test_right_steps_fails_when_the_order_is_wrong():
    case = make_test_case(should_use=["search_help_docs", "create_ticket"])
    trace = make_trace([ok_call("create_ticket"), ok_call("search_help_docs")], reply="Done.")

    result = check_should_use_in_order(case, trace)

    assert not result.passed
    assert "Wrong order" in result.message


def test_right_steps_counts_a_blocked_try_as_a_step():
    case = make_test_case(should_use=["get_ticket"])
    trace = make_trace([blocked_call("get_ticket", {"ticket_id": "TCK-1003"})], reply="Sorry.")

    assert check_should_use_in_order(case, trace).passed


def test_no_should_use_means_no_check():
    case = make_test_case()
    trace = make_trace([ok_call("get_ticket")], reply="Done.")

    assert check_should_use_in_order(case, trace) is None


def test_max_calls_passes_inside_the_limit():
    case = make_test_case(max_calls={"create_ticket": 1})
    trace = make_trace([ok_call("create_ticket")], reply="Done.")

    results = check_max_calls(case, trace)

    assert len(results) == 1
    assert results[0].passed


def test_max_calls_fails_above_the_limit_with_a_clear_message():
    case = make_test_case(max_calls={"create_ticket": 1})
    trace = make_trace(
        [ok_call("create_ticket", {"title": "a"}), ok_call("create_ticket", {"title": "b"})],
        reply="Done.",
    )

    result = check_max_calls(case, trace)[0]

    assert not result.passed
    assert result.message == "Called create_ticket 2 times. Expected at most 1."


def test_max_calls_counts_failed_tries_too():
    case = make_test_case(max_calls={"create_ticket": 1})
    trace = make_trace([failed_call("create_ticket"), failed_call("create_ticket")], reply="Sorry.")

    assert not check_max_calls(case, trace)[0].passed


def test_no_repeated_calls_passes_for_different_inputs():
    case = make_test_case()
    trace = make_trace(
        [ok_call("search_help_docs", {"query": "vpn"}), ok_call("search_help_docs", {"query": "printer"})],
        reply="Done.",
    )

    assert check_no_repeated_calls(case, trace).passed


def test_no_repeated_calls_fails_for_the_same_call_twice():
    case = make_test_case()
    same = {"ticket_id": "TCK-1001"}
    trace = make_trace([ok_call("get_ticket", same), ok_call("get_ticket", same)], reply="Done.")

    result = check_no_repeated_calls(case, trace)

    assert not result.passed
    assert "get_ticket" in result.message


def test_a_failed_retry_is_not_repeated_work():
    case = make_test_case()
    same = {"title": "Laptop", "description": "d", "priority": "high"}
    trace = make_trace([failed_call("create_ticket", same), failed_call("create_ticket", same)], reply="Sorry.")

    assert check_no_repeated_calls(case, trace).passed
