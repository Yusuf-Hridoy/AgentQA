"""Picks the brain for a run.

Phase 1 has one brain: fake. The names local and groq are known, but they arrive
in Phase 2, so asking for them gives a clear message instead of a crash.
"""

from __future__ import annotations

from pathlib import Path

from agentqa.brains.brain_interface import Brain
from agentqa.brains.fake_brain import FakeBrain

BRAIN_FAKE = "fake"
LATER_BRAINS = ("local", "groq")
PHASE_2_MESSAGE = "This brain comes in Phase 2."


class UnknownBrainError(Exception):
    """The brain name is not one AgentQA can use right now."""


def build_brain(brain_name: str, test_id: str, scripts_dir: str | Path | None = None) -> Brain:
    if brain_name == BRAIN_FAKE:
        return FakeBrain.for_test_case(test_id, folder=scripts_dir)
    if brain_name in LATER_BRAINS:
        raise UnknownBrainError(PHASE_2_MESSAGE)
    raise UnknownBrainError(
        "There is no brain called " + str(brain_name) + ". Phase 1 has only: " + BRAIN_FAKE + "."
    )
