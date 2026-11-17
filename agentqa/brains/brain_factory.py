"""Picks the brain for a run, and says early when a brain is not ready.

Phase 2 has four names:

    fake    the scripted brain, used by the unit tests and by CI
    local   Qwen on Ollama, on this machine
    groq    an OpenAI-compatible model on Groq
    gemini  Gemini through its OpenAI-compatible endpoint

check_brain_ready runs once, before the first test. A brain that is not ready
is a setup error, so AgentQA stops with exit code 2 instead of marking ten
tests ERROR for the same reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentqa.brains.brain_errors import BrainSetupError
from agentqa.brains.brain_interface import Brain
from agentqa.brains.brain_presets import BRAIN_LOCAL, REAL_BRAIN_NAMES, preset_for
from agentqa.brains.fake_brain import FakeBrain
from agentqa.brains.openai_compatible_brain import OpenAICompatibleBrain

BRAIN_FAKE = "fake"
ALL_BRAIN_NAMES = (BRAIN_FAKE,) + REAL_BRAIN_NAMES


class UnknownBrainError(Exception):
    """The brain name is not one AgentQA knows."""


def is_real_brain(brain_name: str) -> bool:
    return brain_name in REAL_BRAIN_NAMES


def build_brain(
    brain_name: str,
    test_id: str,
    scripts_dir: str | Path | None = None,
    client: Any = None,
) -> Brain:
    """One brain for one test. Real brains get a fresh one per test."""
    if brain_name == BRAIN_FAKE:
        return FakeBrain.for_test_case(test_id, folder=scripts_dir)
    if is_real_brain(brain_name):
        return OpenAICompatibleBrain(preset_for(brain_name), client=client)
    raise UnknownBrainError(
        "There is no brain called " + str(brain_name)
        + ". Known brains: " + ", ".join(ALL_BRAIN_NAMES) + "."
    )


def brain_model_name(brain_name: str) -> str:
    """The model id this run will use, for the header line and results.json."""
    if brain_name == BRAIN_FAKE:
        return BRAIN_FAKE
    return preset_for(brain_name).model()


def check_brain_ready(
    brain_name: str,
    test_ids: list[str] | None = None,
    scripts_dir: str | Path | None = None,
    client: Any = None,
) -> None:
    """Raise BrainSetupError when this brain cannot run a single test."""
    if brain_name == BRAIN_FAKE:
        for test_id in test_ids or []:
            FakeBrain.for_test_case(test_id, folder=scripts_dir)
        return

    if not is_real_brain(brain_name):
        raise UnknownBrainError(
            "There is no brain called " + str(brain_name)
            + ". Known brains: " + ", ".join(ALL_BRAIN_NAMES) + "."
        )

    preset = preset_for(brain_name)
    # Raises a clear BrainSetupError when the key is missing.
    key = preset.api_key()

    if brain_name != BRAIN_LOCAL:
        return

    check_ollama_ready(preset.model(), base_url=preset.base(), api_key=key, client=client)


def check_ollama_ready(
    model: str,
    base_url: str,
    api_key: str,
    client: Any = None,
) -> None:
    """Ollama must be running, and the model must already be pulled."""
    import openai

    talker = client
    if talker is None:
        talker = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=10, max_retries=0)

    try:
        listed = talker.models.list()
    except Exception as problem:  # noqa: BLE001 - every failure means the same thing here
        raise BrainSetupError(
            "Ollama is not running. Start it with: ollama serve"
        ) from problem

    names = {str(getattr(one, "id", "")) for one in (getattr(listed, "data", None) or [])}
    if not names:
        raise BrainSetupError("Model not found. Run: ollama pull " + model)

    # Ollama answers with the full tag, for example qwen3.5:4b.
    wanted = {model, model + ":latest"}
    if not (names & wanted):
        raise BrainSetupError("Model not found. Run: ollama pull " + model)
