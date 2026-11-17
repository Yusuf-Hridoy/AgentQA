"""Which way a run checks the meaning of a reply.

    words   match reply_should_say, as in Phase 1. Free, exact, no network.
    gemini  ask the AI judge about reply_should_mean.
    auto    words for the fake brain, the judge for a real brain.

auto is the default, so `agentqa run --brain fake` never calls the network and
`agentqa run --brain groq` is judged on meaning without a second flag.
"""

from __future__ import annotations

from agentqa.brains.brain_factory import BRAIN_FAKE, is_real_brain
from agentqa.judge.reply_judge import ReplyJudge

JUDGE_AUTO = "auto"
JUDGE_WORDS = "words"
JUDGE_GEMINI = "gemini"
ALL_JUDGE_MODES = (JUDGE_AUTO, JUDGE_GEMINI, JUDGE_WORDS)


class UnknownJudgeModeError(Exception):
    """The judge mode is not one AgentQA knows."""


def settled_judge_mode(judge_mode: str, brain_name: str) -> str:
    """Turn auto into words or gemini, so the rest of the run has one answer."""
    if judge_mode not in ALL_JUDGE_MODES:
        raise UnknownJudgeModeError(
            "There is no judge mode called " + str(judge_mode)
            + ". Known modes: " + ", ".join(ALL_JUDGE_MODES) + "."
        )
    if judge_mode != JUDGE_AUTO:
        return judge_mode
    if brain_name == BRAIN_FAKE or not is_real_brain(brain_name):
        return JUDGE_WORDS
    return JUDGE_GEMINI


def build_judge(judge_mode: str, use_cache: bool = True, client: object = None) -> ReplyJudge | None:
    """A judge for this run, or None when the run matches words only."""
    if judge_mode != JUDGE_GEMINI:
        return None
    return ReplyJudge(use_cache=use_cache, client=client)
