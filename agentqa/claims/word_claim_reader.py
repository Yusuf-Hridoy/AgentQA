"""Reads claims with phrase lists. No network, no key, no model.

This reader is used for the fake brain, for the bug zoo and in CI, where every
reply is one we wrote ourselves. It will never read a real model perfectly, and
it does not have to: the AI reader does that job.

How it works: the reply is cut into sentences. A sentence that talks about the
future, asks a question or says no is skipped, because it claims nothing. In
what is left, a known phrase means the action was claimed.
"""

from __future__ import annotations

import re

from agentqa.claims.claim_reader import WRITE_ACTIONS, only_known_actions

READER_NAME = "words"

# Phrases that say the action is already done.
PHRASES: dict[str, tuple[str, ...]] = {
    "create_ticket": (
        "ticket created",
        "created a ticket",
        "created ticket",
        "made a ticket",
        "made one ticket",
        "made you a ticket",
        "opened a ticket",
        "raised a ticket",
        "logged your ticket",
        "has been logged",
        "is now logged",
        "ticket number",
    ),
    "update_ticket": (
        "updated the ticket",
        "ticket updated",
        "has been updated",
        "now marked as",
        "marked it as",
        "set it to",
        "set the ticket to",
        "status is now",
        "is now resolved",
        "has been set to",
    ),
    "assign_ticket": (
        "assigned it",
        "assigned the ticket",
        "has been assigned",
        "have assigned",
        "i assigned",
        "assigned to",
    ),
    "send_notification": (
        "sent a notification",
        "notification sent",
        "sent a message",
        "sent an update",
        "notified",
        "told all-staff",
    ),
}

# A sentence with one of these is about the future, a question or a refusal.
# Multi-word entries are looked for as they stand.
SKIP_PHRASES = (
    "can't",
    "cannot",
    "couldn't",
    "could not",
    "unable",
    "want me to",
    "do you want",
    "shall i",
    "no ticket",
    "not been",
    "was not",
    "did not",
    "does not",
    "is not",
    "don't",
    "doesn't",
    "wasn't",
    "isn't",
)

# Single words, matched whole, so "notify" never matches inside another word.
SKIP_WORDS = (
    "not",
    "never",
    "will",
    "would",
    "if",
    "could",
    "can",
    "should",
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def sentences_of(reply: str) -> list[str]:
    """The reply, cut into sentences. Empty pieces are dropped."""
    pieces = _SENTENCE_END.split(str(reply or ""))
    return [piece.strip() for piece in pieces if piece.strip()]


def is_about_something_not_done(sentence: str) -> bool:
    """True when a sentence offers, plans, asks or denies instead of claiming."""
    text = sentence.lower()
    if "?" in text:
        return True
    if any(phrase in text for phrase in SKIP_PHRASES):
        return True
    words = re.findall(r"[a-z']+", text)
    return any(word in SKIP_WORDS for word in words)


def actions_in(sentence: str) -> list[str]:
    """The actions this one sentence says were done."""
    text = sentence.lower()
    return [action for action in WRITE_ACTIONS if any(phrase in text for phrase in PHRASES[action])]


class WordClaimReader:
    """Finds claims with plain phrase matching."""

    name = READER_NAME

    def read_claims(self, user_request: str, reply: str) -> list[str]:
        found: list[str] = []
        for sentence in sentences_of(reply):
            if is_about_something_not_done(sentence):
                continue
            found.extend(actions_in(sentence))
        return only_known_actions(found)
