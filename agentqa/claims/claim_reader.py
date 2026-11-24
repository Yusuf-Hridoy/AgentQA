"""What a claim is, and who reads claims out of a reply.

A claim is one write action the reply says was already done. Only these four
tools change the ticket system, so only these can be claimed:

    create_ticket, update_ticket, assign_ticket, send_notification

Reading the help docs or a ticket changes nothing, so "I checked the docs" is
never a claim. Nor are offers ("I can make a ticket"), plans ("I will open one
if this does not help"), refusals, honest failures or questions.

Two readers follow this one interface:

    word   phrase lists, no network. Used for the fake brain, the bug zoo and CI.
    ai     a model reads the reply. Used for real brains, like the reply judge.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# The only actions a reply can claim. Anything else is dropped.
WRITE_ACTIONS = (
    "create_ticket",
    "update_ticket",
    "assign_ticket",
    "send_notification",
)


@runtime_checkable
class ClaimReader(Protocol):
    """Reads the actions a reply says were already done."""

    name: str

    def read_claims(self, user_request: str, reply: str) -> list[str]:
        ...


def only_known_actions(names: list[str]) -> list[str]:
    """Keep the known write actions, in a fixed order, with no repeats."""
    found = {str(name).strip() for name in names}
    return [action for action in WRITE_ACTIONS if action in found]


def build_claim_reader(
    judge_mode: str,
    use_cache: bool = True,
    client: object = None,
) -> ClaimReader:
    """The reader this run uses. It follows the --judge option.

    words and auto-with-the-fake-brain give the word reader, which needs no
    network. gemini gives the AI reader.
    """
    from agentqa.judge.judge_modes import JUDGE_GEMINI

    if judge_mode == JUDGE_GEMINI:
        from agentqa.claims.ai_claim_reader import AIClaimReader

        return AIClaimReader(use_cache=use_cache, client=client)

    from agentqa.claims.word_claim_reader import WordClaimReader

    return WordClaimReader()
