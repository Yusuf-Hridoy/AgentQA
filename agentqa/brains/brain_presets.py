"""Where each real brain lives: base URL, key name and default model.

All three providers speak the OpenAI chat completions API, so one brain file
works for all of them. Only these few settings differ.

Model ids were looked up on 2026-11-14. See the README for the links.
Every model can be changed with an environment variable, so a new model id
needs no code change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from agentqa.brains.brain_errors import BrainSetupError

BRAIN_LOCAL = "local"
BRAIN_GROQ = "groq"
BRAIN_GEMINI = "gemini"

# Ollama needs a key string but ignores what it says.
LOCAL_PLACEHOLDER_KEY = "ollama"


@dataclass(frozen=True)
class BrainPreset:
    """One provider, as AgentQA talks to it."""

    name: str
    label: str
    base_url: str
    base_url_env: str
    api_key_env: str | None
    default_model: str
    model_env: str
    needs_key: bool
    # Sent with the first request. Dropped and retried once if the provider says no.
    extra_body: dict[str, Any] = field(default_factory=dict)
    send_parallel_tool_calls: bool = True

    def model(self) -> str:
        """The model id for this run: the environment wins, else the default."""
        return os.environ.get(self.model_env, "").strip() or self.default_model

    def base(self) -> str:
        return os.environ.get(self.base_url_env, "").strip() or self.base_url

    def api_key(self) -> str:
        """The key for this provider, or a clear message about where to put it."""
        if not self.needs_key:
            return LOCAL_PLACEHOLDER_KEY
        key = os.environ.get(str(self.api_key_env), "").strip()
        if not key:
            raise BrainSetupError(
                "Set " + str(self.api_key_env) + " in your .env file."
            )
        return key


PRESETS: dict[str, BrainPreset] = {
    BRAIN_LOCAL: BrainPreset(
        name=BRAIN_LOCAL,
        label="Ollama",
        base_url="http://localhost:11434/v1",
        base_url_env="AGENTQA_LOCAL_BASE_URL",
        api_key_env=None,
        # https://ollama.com/library/qwen3.5 - checked 2026-11-14
        default_model="qwen3.5:4b",
        model_env="AGENTQA_LOCAL_MODEL",
        needs_key=False,
        # Qwen thinks out loud by default. Thinking off is much faster.
        extra_body={"think": False},
        send_parallel_tool_calls=False,
    ),
    BRAIN_GROQ: BrainPreset(
        name=BRAIN_GROQ,
        label="Groq",
        base_url="https://api.groq.com/openai/v1",
        base_url_env="AGENTQA_GROQ_BASE_URL",
        api_key_env="GROQ_API_KEY",
        # https://console.groq.com/docs/models - checked 2026-11-14
        default_model="openai/gpt-oss-120b",
        model_env="AGENTQA_GROQ_MODEL",
        needs_key=True,
        send_parallel_tool_calls=True,
    ),
    BRAIN_GEMINI: BrainPreset(
        name=BRAIN_GEMINI,
        label="Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        base_url_env="AGENTQA_GEMINI_BASE_URL",
        api_key_env="GEMINI_API_KEY",
        # https://aistudio.google.com/docs/models - checked 2026-11-14
        default_model="gemini-3.5-flash-lite",
        model_env="AGENTQA_GEMINI_MODEL",
        needs_key=True,
        send_parallel_tool_calls=False,
    ),
}

REAL_BRAIN_NAMES = tuple(PRESETS)


def preset_for(brain_name: str) -> BrainPreset:
    preset = PRESETS.get(brain_name)
    if preset is None:
        raise BrainSetupError(
            "There is no real brain called " + str(brain_name)
            + ". Known: " + ", ".join(REAL_BRAIN_NAMES) + "."
        )
    return preset


def short_model_name(model: str) -> str:
    """gpt-oss-120b, not openai/gpt-oss-120b. Only for the terminal line."""
    return model.split("/")[-1]
