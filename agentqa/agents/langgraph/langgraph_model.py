"""The chat model a LangGraph agent thinks with.

It is built from the brain presets AgentQA already has, so `--brain local` and
`--brain groq` mean the same model, the same base URL and the same key they
mean for the builtin agent. Only the way it is called changes.

Which brains work here:

    fake    the scripted model in langgraph_fake_model.py, for CI
    local   Qwen on Ollama, with thinking off, as the builtin brain has it
    groq    an OpenAI-compatible model on Groq
    gemini  not supported, because of the thought-signature problem. A
            LangGraph run with --brain gemini is a setup error, not ten errors.

A provider failure that is still a failure after the retries becomes a
BrainError, so one test becomes ERROR and the run goes on. That is what the
builtin brain does, and the two have to behave the same or a comparison between
them means nothing.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

import openai
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

from agentqa.agents.agent_interface import AgentSetupError
from agentqa.brains.brain_errors import BrainError
from agentqa.brains.brain_interface import TokenUsage
from agentqa.brains.brain_presets import BRAIN_GEMINI, BRAIN_LOCAL, preset_for
from agentqa.brains.openai_compatible_brain import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    TEMPERATURE,
    strip_thinking,
)

BRAIN_FAKE = "fake"

# The brains a LangGraph agent may use.
LANGGRAPH_BRAINS = (BRAIN_FAKE, BRAIN_LOCAL, "groq")

GEMINI_NOT_SUPPORTED = (
    "Gemini as a LangGraph brain is not supported yet. Use groq or local."
)


class UsageCounter(BaseCallbackHandler):
    """Counts what the model used, from the model's own usage numbers.

    It is a callback and not a read of the final state on purpose: when a run
    stops at the step limit there is no final state, and the tokens the run
    really cost would be lost.
    """

    def __init__(self) -> None:
        self.usage = TokenUsage()
        self._started: dict[str, float] = {}

    # ---------- the callback interface ----------

    def on_llm_start(self, serialized: Any, prompts: Any, *, run_id: UUID = None, **kwargs: Any) -> None:
        self._started[str(run_id)] = time.perf_counter()

    on_chat_model_start = on_llm_start

    def on_llm_end(self, response: LLMResult, *, run_id: UUID = None, **kwargs: Any) -> None:
        started = self._started.pop(str(run_id), None)
        time_ms = (time.perf_counter() - started) * 1000 if started else 0.0
        prompt_tokens, completion_tokens = _tokens_in(response)
        self.usage.add(prompt_tokens, completion_tokens, time_ms)

    def on_llm_error(self, error: BaseException, *, run_id: UUID = None, **kwargs: Any) -> None:
        self._started.pop(str(run_id), None)

    # ---------- what AgentQA asks for ----------

    def totals(self) -> TokenUsage:
        return self.usage.model_copy()


def _tokens_in(response: LLMResult) -> tuple[int, int]:
    """The usage numbers, wherever this provider put them."""
    for batch in response.generations or []:
        for generation in batch:
            message = getattr(generation, "message", None)
            usage = getattr(message, "usage_metadata", None) or {}
            if usage:
                return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)

    totals = (response.llm_output or {}).get("token_usage") or {}
    return int(totals.get("prompt_tokens") or 0), int(totals.get("completion_tokens") or 0)


class AgentQAChatOpenAI(ChatOpenAI):
    """ChatOpenAI that drops the thinking switch once if the provider says no.

    The builtin brain does the same. Ollama takes reasoning_effort and some
    providers refuse it, and losing the whole run over an optional field would
    be silly.
    """

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
        try:
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        except openai.BadRequestError:
            if not self.reasoning_effort:
                raise
            # Dropped for the rest of the run, then asked once more.
            self.reasoning_effort = None
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def check_brain_supported(brain_name: str) -> None:
    """Stop before the first test when this brain cannot drive a LangGraph agent."""
    if brain_name == BRAIN_GEMINI:
        raise AgentSetupError(GEMINI_NOT_SUPPORTED)
    if brain_name not in LANGGRAPH_BRAINS:
        raise AgentSetupError(
            "There is no LangGraph brain called " + str(brain_name)
            + ". Known: " + ", ".join(LANGGRAPH_BRAINS) + "."
        )


def build_chat_model(brain_name: str, model: str | None = None) -> Any:
    """The chat model for this brain, built from the preset AgentQA already has."""
    check_brain_supported(brain_name)
    preset = preset_for(brain_name)
    return AgentQAChatOpenAI(
        model=model or preset.model(),
        base_url=preset.base(),
        api_key=preset.api_key(),
        temperature=TEMPERATURE,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
        # Qwen on Ollama thinks out loud by default and then answers with an
        # empty reply. "none" turns that off, the way the builtin brain does.
        reasoning_effort=preset.reasoning_effort,
        # These providers speak chat completions. The responses API is not what
        # the builtin brain uses, and the two must be comparable.
        use_responses_api=False,
    )


def as_brain_error(problem: BaseException) -> BrainError:
    """One provider failure, as the error that costs one test and not the run."""
    message = str(problem) or problem.__class__.__name__
    return BrainError(message)


def reply_text_of(message: Any) -> str:
    """The text of a final model message, with any <think> block cut out."""
    content = getattr(message, "content", "") or ""
    if isinstance(content, list):
        # Some providers answer in parts. Keep the text parts, in order.
        pieces = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                pieces.append(str(part.get("text") or ""))
        content = "".join(pieces)
    return strip_thinking(str(content))


def finish_reason_of(message: Any) -> str | None:
    """Why the provider stopped, when it said. Empty for the fake model."""
    data = getattr(message, "response_metadata", None) or {}
    reason = data.get("finish_reason") or data.get("stop_reason")
    return str(reason) if reason else None
