"""The AI judge, with a fake client. No network, no key."""

from __future__ import annotations

import json

import pytest
from conftest import make_test_case
from fake_openai_client import FakeOpenAIClient, text_answer

from agentqa.checks.check_result import SOURCE_CODE, SOURCE_JUDGE
from agentqa.checks.reply_checks import JUDGE_CHECK_NAME, run_reply_checks
from agentqa.judge.judge_cache import JudgeCache, cache_key
from agentqa.judge.judge_modes import (
    JUDGE_AUTO,
    JUDGE_GEMINI,
    JUDGE_WORDS,
    UnknownJudgeModeError,
    settled_judge_mode,
)
from agentqa.judge.judge_prompt import JUDGE_PROMPT_VERSION
from agentqa.judge.reply_judge import JudgeError, ReplyJudge

REQUEST = "What is happening with my ticket TCK-1001?"
EXPECTATION = "Tells the user that ticket TCK-1001 is in progress."


def verdict_answer(verdict: str, reason: str = "it says so"):
    return text_answer(json.dumps({"verdict": verdict, "reason": reason}), 40, 12)


def make_judge(answers, tmp_path, use_cache=True):
    client = FakeOpenAIClient(answers=answers)
    cache = JudgeCache(path=tmp_path / "judge_cache.json", enabled=use_cache)
    judge = ReplyJudge(model="test-judge", client=client, cache=cache)
    return judge, client


# ---------- reading the verdict ----------


def test_a_pass_verdict_is_read(tmp_path):
    judge, _ = make_judge([verdict_answer("pass", "the reply gives the status")], tmp_path)

    verdict = judge.judge(REQUEST, "TCK-1001 is in progress.", EXPECTATION)

    assert verdict.passed
    assert verdict.reason == "the reply gives the status"


def test_a_fail_verdict_is_read(tmp_path):
    judge, _ = make_judge([verdict_answer("fail", "the reply gives no status")], tmp_path)

    verdict = judge.judge(REQUEST, "I opened the ticket.", EXPECTATION)

    assert not verdict.passed
    assert verdict.reason == "the reply gives no status"


def test_json_inside_a_code_fence_is_still_read(tmp_path):
    judge, _ = make_judge([text_answer('```json\n{"verdict": "pass", "reason": "fine"}\n```')], tmp_path)

    assert judge.judge(REQUEST, "In progress.", EXPECTATION).passed


# ---------- broken answers ----------


def test_broken_json_is_asked_once_more_and_then_works(tmp_path):
    judge, client = make_judge(
        [text_answer("I think it is fine, really."), verdict_answer("pass", "fine")],
        tmp_path,
    )

    verdict = judge.judge(REQUEST, "In progress.", EXPECTATION)

    assert verdict.passed
    assert client.call_count == 2


def test_broken_json_twice_is_a_judge_error(tmp_path):
    judge, client = make_judge(
        [text_answer("no json here"), text_answer("still no json")],
        tmp_path,
    )

    with pytest.raises(JudgeError):
        judge.judge(REQUEST, "In progress.", EXPECTATION)

    assert client.call_count == 2


def test_a_verdict_that_is_not_pass_or_fail_is_a_judge_error(tmp_path):
    judge, _ = make_judge(
        [verdict_answer("maybe"), verdict_answer("unsure")],
        tmp_path,
    )

    with pytest.raises(JudgeError):
        judge.judge(REQUEST, "In progress.", EXPECTATION)


# ---------- the cache ----------


def test_the_same_reply_is_judged_only_once(tmp_path):
    judge, client = make_judge([verdict_answer("pass", "fine")], tmp_path)

    first = judge.judge(REQUEST, "In progress.", EXPECTATION)
    second = judge.judge(REQUEST, "In progress.", EXPECTATION)

    assert first.passed and second.passed
    assert not first.from_cache and second.from_cache
    assert client.call_count == 1


def test_the_cache_is_saved_and_read_back(tmp_path):
    judge, _ = make_judge([verdict_answer("pass", "fine")], tmp_path)
    judge.judge(REQUEST, "In progress.", EXPECTATION)

    later_client = FakeOpenAIClient(answers=[])
    later = ReplyJudge(
        model="test-judge",
        client=later_client,
        cache=JudgeCache(path=tmp_path / "judge_cache.json"),
    )
    verdict = later.judge(REQUEST, "In progress.", EXPECTATION)

    assert verdict.passed and verdict.from_cache
    assert later_client.call_count == 0


def test_no_cache_asks_the_judge_every_time(tmp_path):
    judge, client = make_judge(
        [verdict_answer("pass", "fine"), verdict_answer("pass", "fine")],
        tmp_path,
        use_cache=False,
    )

    judge.judge(REQUEST, "In progress.", EXPECTATION)
    judge.judge(REQUEST, "In progress.", EXPECTATION)

    assert client.call_count == 2


def test_the_cache_key_changes_with_the_prompt_version_and_the_reply():
    base = cache_key("m", JUDGE_PROMPT_VERSION, REQUEST, "a reply", EXPECTATION)

    assert base != cache_key("m", "judge-v2", REQUEST, "a reply", EXPECTATION)
    assert base != cache_key("m", JUDGE_PROMPT_VERSION, REQUEST, "another reply", EXPECTATION)
    assert base != cache_key("other", JUDGE_PROMPT_VERSION, REQUEST, "a reply", EXPECTATION)
    assert base == cache_key("m", JUDGE_PROMPT_VERSION, REQUEST, "a reply", EXPECTATION)


def test_a_broken_cache_file_is_ignored(tmp_path):
    path = tmp_path / "judge_cache.json"
    path.write_text("this is not json", encoding="utf-8")

    cache = JudgeCache(path=path)

    assert cache.get("anything") is None


# ---------- judge tokens ----------


def test_judge_tokens_are_counted(tmp_path):
    judge, _ = make_judge([verdict_answer("pass", "fine")], tmp_path)

    judge.judge(REQUEST, "In progress.", EXPECTATION)
    usage = judge.usage_totals()

    assert usage.prompt_tokens == 40
    assert usage.completion_tokens == 12
    assert usage.calls == 1


# ---------- how the checks use the judge ----------


def test_the_judge_replaces_the_word_check(tmp_path):
    case = make_test_case(
        user_asks=REQUEST,
        reply_should_say=["in progress"],
        reply_should_mean=EXPECTATION,
    )
    judge, _ = make_judge([verdict_answer("pass", "the reply gives the status")], tmp_path)

    results = run_reply_checks(case, "Your ticket is still being worked on.", judge=judge)
    names = [check.check_name for check in results]

    assert JUDGE_CHECK_NAME in names
    assert "reply_says_the_right_things" not in names
    judged = [check for check in results if check.check_name == JUDGE_CHECK_NAME][0]
    assert judged.passed and judged.source == SOURCE_JUDGE


def test_a_judged_failure_shows_the_reason(tmp_path):
    case = make_test_case(user_asks=REQUEST, reply_should_mean=EXPECTATION)
    judge, _ = make_judge([verdict_answer("fail", "the reply gives no status")], tmp_path)

    results = run_reply_checks(case, "I opened the ticket.", judge=judge)
    judged = [check for check in results if check.check_name == JUDGE_CHECK_NAME][0]

    assert not judged.passed
    assert judged.message == "judge: the reply gives no status"
    assert judged.reason == "the reply gives no status"


def test_without_reply_should_mean_the_words_are_used(tmp_path):
    case = make_test_case(reply_should_say=["ticket"])
    judge, client = make_judge([], tmp_path)

    results = run_reply_checks(case, "I made a ticket.", judge=judge)
    names = [check.check_name for check in results]

    assert "reply_says_the_right_things" in names
    assert JUDGE_CHECK_NAME not in names
    assert client.call_count == 0


def test_without_a_judge_the_words_are_used():
    case = make_test_case(reply_should_say=["ticket"], reply_should_mean=EXPECTATION)

    results = run_reply_checks(case, "I made a ticket.", judge=None)
    names = [check.check_name for check in results]

    assert "reply_says_the_right_things" in names
    assert all(check.source == SOURCE_CODE for check in results)


def test_an_empty_reply_fails_without_asking_the_judge(tmp_path):
    case = make_test_case(reply_should_mean=EXPECTATION)
    judge, client = make_judge([], tmp_path)

    results = run_reply_checks(case, None, judge=judge)

    assert not all(check.passed for check in results)
    assert client.call_count == 0


# ---------- judge modes ----------


def test_auto_uses_words_for_the_fake_brain():
    assert settled_judge_mode(JUDGE_AUTO, "fake") == JUDGE_WORDS


@pytest.mark.parametrize("brain", ["local", "groq", "gemini"])
def test_auto_uses_the_judge_for_a_real_brain(brain):
    assert settled_judge_mode(JUDGE_AUTO, brain) == JUDGE_GEMINI


def test_words_and_gemini_are_kept_as_asked():
    assert settled_judge_mode(JUDGE_WORDS, "groq") == JUDGE_WORDS
    assert settled_judge_mode(JUDGE_GEMINI, "fake") == JUDGE_GEMINI


def test_an_unknown_judge_mode_is_refused():
    with pytest.raises(UnknownJudgeModeError):
        settled_judge_mode("vibes", "fake")
