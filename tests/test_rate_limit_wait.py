"""Waiting out a rate limit. No test ever really sleeps."""

from __future__ import annotations

import httpx
import openai
import pytest
from fake_openai_client import FakeOpenAIClient, text_answer

from agentqa.brains.brain_errors import BrainError
from agentqa.brains.brain_interface import FinalReply
from agentqa.brains.brain_presets import BRAIN_GROQ, preset_for
from agentqa.brains.openai_compatible_brain import OpenAICompatibleBrain
from agentqa.brains.rate_limit_wait import MAX_WAIT_SECONDS, seconds_to_wait
from agentqa.fake_ticket_system.it_support_tools import tool_specs
from agentqa.judge.judge_cache import JudgeCache
from agentqa.judge.reply_judge import JudgeError, ReplyJudge

PRESET = preset_for(BRAIN_GROQ)
TOOLS = tool_specs(["search_help_docs"])


class Clock:
    """Remembers what it was asked to wait for, and waits for nothing."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def rate_limit(message: str = "slow down", body=None, headers=None):
    answer = httpx.Response(
        429,
        request=httpx.Request("POST", "https://example.invalid/v1"),
        headers=headers or {},
    )
    return openai.RateLimitError(message, response=answer, body=body)


# ---------- reading how long to wait ----------


def test_the_wait_is_read_from_a_retry_after_header():
    assert seconds_to_wait(rate_limit(headers={"retry-after": "12"})) == 12.0


def test_the_wait_is_read_from_googles_message():
    problem = rate_limit(
        body=[{"error": {"code": 429, "message": "Quota exceeded. Please retry in 17.75581561s."}}]
    )

    assert seconds_to_wait(problem) == pytest.approx(17.7558, abs=0.001)


def test_the_wait_is_read_from_a_retry_delay_field():
    problem = rate_limit(body={"error": {"details": [{"retryDelay": "23s"}]}})

    assert seconds_to_wait(problem) == 23.0


def test_the_wait_is_read_from_groqs_wording():
    problem = rate_limit(body={"error": {"message": "Rate limit reached. Please try again in 7.5s"}})

    assert seconds_to_wait(problem) == 7.5


def test_milliseconds_are_understood():
    assert seconds_to_wait(rate_limit(headers={"retry-after-ms": "2500"})) == 2.5


def test_a_very_long_wait_is_cut_to_a_minute():
    problem = rate_limit(body={"error": {"message": "Please retry in 3600s"}})

    assert seconds_to_wait(problem) == MAX_WAIT_SECONDS


def test_no_wait_is_given_when_the_provider_did_not_say():
    assert seconds_to_wait(rate_limit("Too many requests.")) is None


# ---------- the brain waits, then tries once more ----------


def test_the_brain_waits_the_asked_time_and_then_succeeds():
    clock = Clock()
    client = FakeOpenAIClient(
        answers=[
            rate_limit(body={"error": {"message": "Please retry in 18s"}}),
            text_answer("I made a ticket."),
        ]
    )
    brain = OpenAICompatibleBrain(PRESET, model="test-model", client=client, sleeper=clock)

    action = brain.next_action([], TOOLS)

    assert isinstance(action, FinalReply)
    assert clock.waits == [18.0]
    assert client.call_count == 2


def test_a_second_rate_limit_ends_that_one_test():
    clock = Clock()
    client = FakeOpenAIClient(
        answers=[
            rate_limit(body={"error": {"message": "Please retry in 5s"}}),
            rate_limit(body={"error": {"message": "Please retry in 5s"}}),
        ]
    )
    brain = OpenAICompatibleBrain(PRESET, model="test-model", client=client, sleeper=clock)

    with pytest.raises(BrainError) as problem:
        brain.next_action([], TOOLS)

    assert "rate limit" in str(problem.value)
    assert clock.waits == [5.0]
    assert client.call_count == 2


def test_the_brain_does_not_wait_when_no_time_was_given():
    clock = Clock()
    client = FakeOpenAIClient(answers=[rate_limit("Too many requests.")])
    brain = OpenAICompatibleBrain(PRESET, model="test-model", client=client, sleeper=clock)

    with pytest.raises(BrainError):
        brain.next_action([], TOOLS)

    assert clock.waits == []
    assert client.call_count == 1


def test_waiting_never_hides_a_different_failure():
    clock = Clock()
    answer = httpx.Response(400, request=httpx.Request("POST", "https://example.invalid/v1"))
    client = FakeOpenAIClient(answers=[openai.BadRequestError("bad", response=answer, body=None)])
    brain = OpenAICompatibleBrain(PRESET, model="test-model", client=client, sleeper=clock)
    brain._send_parallel_flag = False
    brain._extra_body = {}

    with pytest.raises(BrainError):
        brain.next_action([], TOOLS)

    assert clock.waits == []


# ---------- the judge waits too ----------


def test_the_judge_waits_the_asked_time_and_then_succeeds(tmp_path):
    clock = Clock()
    client = FakeOpenAIClient(
        answers=[
            rate_limit(body={"error": {"message": "Please retry in 9s"}}),
            text_answer('{"verdict": "pass", "reason": "fine"}'),
        ]
    )
    judge = ReplyJudge(
        model="test-judge",
        client=client,
        cache=JudgeCache(path=tmp_path / "c.json", enabled=False),
        sleeper=clock,
    )

    verdict = judge.judge("q", "a reply", "an expectation")

    assert verdict.passed
    assert clock.waits == [9.0]


def test_a_second_rate_limit_makes_the_judge_give_up(tmp_path):
    clock = Clock()
    client = FakeOpenAIClient(
        answers=[
            rate_limit(body={"error": {"message": "Please retry in 4s"}}),
            rate_limit(body={"error": {"message": "Please retry in 4s"}}),
        ]
    )
    judge = ReplyJudge(
        model="test-judge",
        client=client,
        cache=JudgeCache(path=tmp_path / "c.json", enabled=False),
        sleeper=clock,
    )

    with pytest.raises(JudgeError) as problem:
        judge.judge("q", "a reply", "an expectation")

    assert "rate limit" in str(problem.value)
    assert clock.waits == [4.0]
