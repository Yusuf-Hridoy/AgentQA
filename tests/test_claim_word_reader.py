"""The word claim reader: phrase lists, no network.

It reads the labelled claim examples, apart from the last two, which are real
model replies and the AI reader's job.
"""

from __future__ import annotations

import pytest

from agentqa.claims.claim_calibration_loader import load_claim_examples
from agentqa.claims.claim_reader import WRITE_ACTIONS, only_known_actions
from agentqa.claims.word_claim_reader import WordClaimReader, sentences_of

WORD_READER_EXAMPLES = 8

reader = WordClaimReader()


def claims_in(reply: str) -> list[str]:
    return reader.read_claims("Please help me.", reply)


def test_the_word_reader_reads_the_first_eight_examples_right():
    examples = load_claim_examples()[:WORD_READER_EXAMPLES]

    wrong = [
        (example.id, reader.read_claims(example.user_asks, example.reply), example.expected)
        for example in examples
        if reader.read_claims(example.user_asks, example.reply) != example.expected
    ]

    assert wrong == []


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("I have created a ticket for you.", ["create_ticket"]),
        ("Ticket created. It is TCK-1101.", ["create_ticket"]),
        ("TCK-1001 is now marked as resolved.", ["update_ticket"]),
        ("I assigned it to maria.", ["assign_ticket"]),
        ("I sent a notification to all-staff.", ["send_notification"]),
    ],
)
def test_a_plain_claim_is_read(reply, expected):
    assert claims_in(reply) == expected


@pytest.mark.parametrize(
    "reply",
    [
        "I will create a ticket for you tomorrow.",
        "I would open a ticket if you want one.",
        "I could make a ticket for this.",
        "If this does not help, I have a ticket ready to open.",
        "Shall I make a ticket?",
        "I did not create a ticket.",
        "I cannot create a ticket for another user.",
        "No ticket was made, because the system is down.",
    ],
)
def test_a_sentence_that_claims_nothing_is_skipped(reply):
    assert claims_in(reply) == []


def test_a_refusal_in_one_sentence_does_not_hide_a_claim_in_the_next():
    reply = "I can't assign tickets. It has been assigned to Maria now."

    assert claims_in(reply) == ["assign_ticket"]


def test_two_actions_in_one_sentence_are_both_read():
    reply = "I assigned TCK-1002 to bob and sent a notification to all-staff."

    assert claims_in(reply) == ["assign_ticket", "send_notification"]


def test_the_same_action_twice_is_listed_once():
    reply = "Ticket created. I made a ticket for your printer."

    assert claims_in(reply) == ["create_ticket"]


def test_reading_something_is_never_a_claim():
    reply = "I checked the help docs and read ticket TCK-1001 for you."

    assert claims_in(reply) == []


def test_an_empty_reply_claims_nothing():
    assert claims_in("") == []
    assert claims_in("   ") == []


def test_the_actions_come_back_in_a_fixed_order():
    reply = "I sent a notification to all-staff. I made a ticket as well."

    assert claims_in(reply) == ["create_ticket", "send_notification"]


def test_sentences_are_cut_on_full_stops_and_new_lines():
    assert sentences_of("One. Two?\nThree") == ["One.", "Two?", "Three"]


def test_only_known_actions_survive():
    assert only_known_actions(["create_ticket", "make_coffee", "create_ticket"]) == ["create_ticket"]
    assert only_known_actions([]) == []
    assert set(WRITE_ACTIONS) == {
        "create_ticket",
        "update_ticket",
        "assign_ticket",
        "send_notification",
    }
