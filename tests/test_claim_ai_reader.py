"""The AI claim reader, with a fake client. No network, no key."""

from __future__ import annotations

import pytest
from fake_openai_client import FakeOpenAIClient, text_answer

from agentqa.brains.brain_presets import BRAIN_GEMINI
from agentqa.claims.ai_claim_reader import AIClaimReader, read_claimed_actions
from agentqa.claims.claim_prompt import CLAIM_PROMPT_VERSION
from agentqa.judge.judge_cache import JudgeCache, cache_key
from agentqa.judge.judge_prompt import JUDGE_PROMPT_VERSION
from agentqa.judge.reply_judge import JudgeError

REQUEST = "My printer is not printing. Please make a ticket."
REPLY = "I made a ticket for your printer. It is TCK-1101."


def make_reader(answers, cache=None):
    client = FakeOpenAIClient(answers=answers)
    reader = AIClaimReader(
        model="test-judge",
        client=client,
        cache=cache if cache is not None else JudgeCache(enabled=False),
        brain_name=BRAIN_GEMINI,
    )
    return reader, client


def test_a_good_answer_gives_the_claimed_actions():
    reader, client = make_reader([text_answer('{"claimed_actions": ["create_ticket"]}')])

    assert reader.read_claims(REQUEST, REPLY) == ["create_ticket"]
    assert client.call_count == 1


def test_an_empty_list_means_the_reply_claims_nothing():
    reader, _ = make_reader([text_answer('{"claimed_actions": []}')])

    assert reader.read_claims(REQUEST, "If this does not help, I can open a ticket.") == []


def test_names_outside_the_fixed_list_are_dropped():
    reader, _ = make_reader(
        [text_answer('{"claimed_actions": ["create_ticket", "make_coffee", "search_help_docs"]}')]
    )

    assert reader.read_claims(REQUEST, REPLY) == ["create_ticket"]


def test_broken_json_is_asked_again_and_then_works():
    reader, client = make_reader(
        [text_answer("I think it made a ticket."), text_answer('{"claimed_actions": ["create_ticket"]}')]
    )

    assert reader.read_claims(REQUEST, REPLY) == ["create_ticket"]
    assert client.call_count == 2


def test_broken_json_twice_raises_a_judge_error():
    reader, client = make_reader([text_answer("no json"), text_answer("still no json")])

    with pytest.raises(JudgeError):
        reader.read_claims(REQUEST, REPLY)
    assert client.call_count == 2


def test_an_empty_answer_raises_a_judge_error():
    reader, _ = make_reader([text_answer("ignored")])
    reader.client.answers[0].choices = []

    with pytest.raises(JudgeError):
        reader.read_claims(REQUEST, REPLY)


def test_an_empty_reply_is_never_sent_to_the_model():
    reader, client = make_reader([])

    assert reader.read_claims(REQUEST, "   ") == []
    assert client.call_count == 0


def test_a_cached_answer_costs_no_call(tmp_path):
    cache = JudgeCache(path=tmp_path / "cache.json", enabled=True)
    first, first_client = make_reader([text_answer('{"claimed_actions": ["create_ticket"]}')], cache=cache)
    first.read_claims(REQUEST, REPLY)

    second, second_client = make_reader([], cache=cache)

    assert second.read_claims(REQUEST, REPLY) == ["create_ticket"]
    assert first_client.call_count == 1
    assert second_client.call_count == 0


def test_claims_and_judge_answers_never_share_a_cache_key():
    claims = cache_key("m", CLAIM_PROMPT_VERSION, REQUEST, REPLY, "", label="claims")
    judged = cache_key("m", JUDGE_PROMPT_VERSION, REQUEST, REPLY, "")

    assert claims != judged


def test_the_tokens_are_counted():
    reader, _ = make_reader([text_answer('{"claimed_actions": []}', prompt_tokens=11, completion_tokens=3)])

    reader.read_claims(REQUEST, REPLY)

    assert reader.usage_totals().prompt_tokens == 11
    assert reader.usage_totals().completion_tokens == 3
    assert reader.usage_totals().calls == 1


def test_reading_an_answer_without_the_list_is_an_error():
    with pytest.raises(JudgeError):
        read_claimed_actions('{"verdict": "pass"}')
    with pytest.raises(JudgeError):
        read_claimed_actions('{"claimed_actions": "create_ticket"}')


def test_a_fenced_answer_is_still_read():
    assert read_claimed_actions('```json\n{"claimed_actions": ["update_ticket"]}\n```') == ["update_ticket"]
