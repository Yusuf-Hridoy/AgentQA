"""The check that runs before the first test. No network, no key."""

from __future__ import annotations

import pytest
from fake_openai_client import FakeOpenAIClient, model_list

from agentqa.brains.brain_errors import BrainSetupError
from agentqa.brains.brain_factory import (
    BRAIN_FAKE,
    UnknownBrainError,
    brain_model_name,
    check_brain_ready,
    check_ollama_ready,
    is_real_brain,
)
from agentqa.brains.brain_presets import BRAIN_GEMINI, BRAIN_GROQ, BRAIN_LOCAL, preset_for


def test_a_missing_groq_key_says_where_to_put_it(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(BrainSetupError) as problem:
        check_brain_ready(BRAIN_GROQ)

    assert str(problem.value) == "Set GROQ_API_KEY in your .env file."


def test_a_missing_gemini_key_says_where_to_put_it(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(BrainSetupError) as problem:
        check_brain_ready(BRAIN_GEMINI)

    assert str(problem.value) == "Set GEMINI_API_KEY in your .env file."


def test_a_key_that_is_set_passes_the_check(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "a-test-key")

    check_brain_ready(BRAIN_GROQ)


def test_the_fake_brain_is_ready_when_its_script_exists():
    check_brain_ready(BRAIN_FAKE, ["printer-ticket"])


def test_an_unknown_brain_name_is_refused():
    with pytest.raises(UnknownBrainError) as problem:
        check_brain_ready("magic")

    assert "no brain called magic" in str(problem.value)


def test_ollama_that_does_not_answer_says_how_to_start_it():
    client = FakeOpenAIClient(models=ConnectionError("connection refused"))

    with pytest.raises(BrainSetupError) as problem:
        check_ollama_ready("qwen3.5:4b", "http://localhost:11434/v1", "ollama", client=client)

    assert str(problem.value) == "Ollama is not running. Start it with: ollama serve"


def test_a_model_that_is_not_pulled_says_how_to_pull_it():
    client = FakeOpenAIClient(models=model_list("llama3:8b"))

    with pytest.raises(BrainSetupError) as problem:
        check_ollama_ready("qwen3.5:4b", "http://localhost:11434/v1", "ollama", client=client)

    assert str(problem.value) == "Model not found. Run: ollama pull qwen3.5:4b"


def test_a_pulled_model_passes_the_check():
    client = FakeOpenAIClient(models=model_list("qwen3.5:4b", "llama3:8b"))

    check_ollama_ready("qwen3.5:4b", "http://localhost:11434/v1", "ollama", client=client)


def test_the_model_id_can_be_changed_with_an_environment_variable(monkeypatch):
    monkeypatch.setenv("AGENTQA_GROQ_MODEL", "openai/gpt-oss-20b")

    assert brain_model_name(BRAIN_GROQ) == "openai/gpt-oss-20b"


def test_each_preset_has_a_base_url_and_a_default_model():
    for name in (BRAIN_LOCAL, BRAIN_GROQ, BRAIN_GEMINI):
        preset = preset_for(name)
        assert preset.base().startswith("http")
        assert preset.default_model
        assert is_real_brain(name)

    assert not is_real_brain(BRAIN_FAKE)
    assert brain_model_name(BRAIN_FAKE) == BRAIN_FAKE


def test_the_local_brain_needs_no_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    assert preset_for(BRAIN_LOCAL).api_key() == "ollama"
