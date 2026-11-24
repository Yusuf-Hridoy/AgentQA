"""A prompt must never quote the examples it is later scored on.

`agentqa check-judge` measures the judge and the claim reader against the
hand-labelled replies in judge_calibration/. If one of those replies is written
into a prompt, the model is told the answer and the score stops meaning
anything. This test fails when any run of words from a calibration reply turns
up inside the judge prompt or the claim prompt.
"""

from __future__ import annotations

import re

import pytest

from agentqa.claims.claim_calibration_loader import load_claim_examples
from agentqa.claims.claim_prompt import CLAIM_SYSTEM_PROMPT
from agentqa.judge.judge_calibration_loader import load_judge_examples
from agentqa.judge.judge_prompt import JUDGE_SYSTEM_PROMPT

# Four words in a row. Shorter runs catch ordinary English ("the ticket was"),
# longer runs let a whole quoted clause through.
RUN_OF_WORDS = 4


def plain_words(text: str) -> list[str]:
    """The text as plain lowercase words, so punctuation cannot hide a quote."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split()


def runs_of_words(text: str, length: int = RUN_OF_WORDS) -> set[str]:
    words = plain_words(text)
    return {" ".join(words[at:at + length]) for at in range(len(words) - length + 1)}


def quoted_in(prompt: str, reply: str) -> list[str]:
    """Every run of words the reply and the prompt share, worst first."""
    haystack = " " + " ".join(plain_words(prompt)) + " "
    found = [run for run in runs_of_words(reply) if " " + run + " " in haystack]
    return sorted(found)


def calibration_replies() -> list[tuple[str, str]]:
    pairs = [(example.id, example.reply) for example in load_judge_examples()]
    pairs += [(example.id, example.reply) for example in load_claim_examples()]
    return pairs


PROMPTS = [("judge prompt", JUDGE_SYSTEM_PROMPT), ("claim prompt", CLAIM_SYSTEM_PROMPT)]


@pytest.mark.parametrize("prompt_name,prompt", PROMPTS)
def test_no_calibration_reply_is_quoted_in_a_prompt(prompt_name, prompt):
    leaks = []
    for example_id, reply in calibration_replies():
        shared = quoted_in(prompt, reply)
        if shared:
            leaks.append(example_id + ": " + ", ".join('"' + run + '"' for run in shared))

    assert not leaks, (
        "The " + prompt_name + " quotes calibration replies, so check-judge would be "
        "measuring words the model was handed:\n  " + "\n  ".join(leaks)
    )


def test_the_check_notices_a_quote_that_is_really_there():
    """The guard above is only worth having if it can fail."""
    example_id, reply = calibration_replies()[0]
    assert quoted_in("A prompt that says: " + reply, reply), example_id


def test_a_prompt_that_shares_only_short_phrases_is_allowed():
    """Three words in common is ordinary English, not a quoted example."""
    assert quoted_in("I made a note about the thing.", "I made a ticket for you.") == []
