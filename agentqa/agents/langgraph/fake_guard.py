"""A guard with no model, for --brain fake.

It is a word rule, not a thinker. It exists so CI can run the guard agent with
no key and no network, and so the unit tests can set up an exact situation and
know what the guard will say.

The rule: a write is allowed when the user's own request holds a word that
matches the action. Anything else is blocked. That is enough to keep the ten
fake scripts passing and to make the guard's behaviour testable, and it is not
a model, so never read a fake-brain guard result as evidence about a real one.
"""

from __future__ import annotations

import re

from agentqa.agents.agent_interface import GuardDecision

# Which words in the user's request stand for which action.
WORDS_FOR_ACTION: dict[str, tuple[str, ...]] = {
    "create_ticket": ("ticket", "create", "make", "open", "raise", "log"),
    "update_ticket": ("update", "mark", "change", "set", "resolve", "resolved", "close", "closed"),
    "assign_ticket": ("assign", "assigned"),
    "send_notification": ("tell", "notify", "message", "know", "email"),
}

ALLOWED_REASON = "The user asked for this."
BLOCKED_REASON = "The user did not ask for this."


def plain_words(text: str) -> set[str]:
    return set(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())


class FakeGuard:
    """Allows a write only when the user's request has a matching word."""

    name = "fake"

    def decide(self, user_asks: str, tool_name: str, inputs: dict) -> GuardDecision:
        wanted = WORDS_FOR_ACTION.get(tool_name, ())
        said = plain_words(user_asks)
        if said & set(wanted):
            return GuardDecision(allow=True, reason=ALLOWED_REASON)
        return GuardDecision(allow=False, reason=BLOCKED_REASON)

    def usage_totals(self):
        from agentqa.brains.brain_interface import TokenUsage

        return TokenUsage()


class BlockEverythingGuard:
    """A guard that refuses every write. Only the unit tests use it.

    It is how a wrongly blocked action is produced on purpose, so the guard
    accounting can be tested without waiting for a model to make a mistake.
    """

    name = "block-everything"

    def __init__(self, reason: str = BLOCKED_REASON) -> None:
        self.reason = reason

    def decide(self, user_asks: str, tool_name: str, inputs: dict) -> GuardDecision:
        return GuardDecision(allow=False, reason=self.reason)

    def usage_totals(self):
        from agentqa.brains.brain_interface import TokenUsage

        return TokenUsage()
