"""The AI claim reader: what does this reply say the agent already did?

It is set up like the reply judge, because it does the same kind of work: one
small question about one reply, at temperature 0, cached, with its tokens
counted as judge tokens.

It never decides whether a claim is true. That is a fact, so claim_vs_reality
checks it against the trace in plain code.

A broken answer is tried once more. If it is still broken, a JudgeError is
raised, which marks the test ERROR, exactly as the judge does.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import openai

from agentqa.brains.brain_interface import TokenUsage
from agentqa.brains.brain_presets import BRAIN_GEMINI, preset_for
from agentqa.brains.rate_limit_wait import seconds_to_wait
from agentqa.claims.claim_prompt import (
    CLAIM_PROMPT_VERSION,
    CLAIM_SYSTEM_PROMPT,
    RETRY_REMINDER,
    build_claim_question,
)
from agentqa.claims.claim_reader import only_known_actions
from agentqa.judge.judge_cache import JudgeCache, cache_key
from agentqa.judge.reply_judge import JudgeError

READER_NAME = "ai"
CACHE_LABEL = "claims"
CLAIM_TIMEOUT_SECONDS = 60
CLAIM_MAX_RETRIES = 3
CLAIM_TEMPERATURE = 0.0

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)


def read_claimed_actions(text: str) -> list[str]:
    """Pull the action list out of the answer. Raises when it is not there."""
    cleaned = _FENCE.sub("", str(text or "")).strip()
    found = _JSON_OBJECT.search(cleaned)
    if not found:
        raise JudgeError("The claim reader did not answer with JSON.")
    try:
        answer = json.loads(found.group(0))
    except ValueError as problem:
        raise JudgeError("The claim reader answered with broken JSON.") from problem
    if not isinstance(answer, dict) or "claimed_actions" not in answer:
        raise JudgeError("The claim reader gave no claimed_actions list.")

    listed = answer.get("claimed_actions")
    if not isinstance(listed, list):
        raise JudgeError("claimed_actions was not a list.")
    # Anything not on the fixed list is dropped, without a complaint.
    return only_known_actions([str(one) for one in listed])


class AIClaimReader:
    """Asks a model what one reply says the agent already did."""

    name = READER_NAME

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        client: Any = None,
        cache: JudgeCache | None = None,
        use_cache: bool = True,
        brain_name: str = BRAIN_GEMINI,
        sleeper: Any = None,
    ) -> None:
        self.preset = preset_for(brain_name)
        # Swapped out in the unit tests, so no test ever really waits.
        self._sleep = sleeper if sleeper is not None else time.sleep
        self.model = model or self.preset.model()
        self.prompt_version = CLAIM_PROMPT_VERSION
        self.cache = cache if cache is not None else JudgeCache(enabled=use_cache)
        self._usage = TokenUsage()

        if client is not None:
            self.client = client
        else:
            self.client = openai.OpenAI(
                base_url=self.preset.base(),
                api_key=api_key if api_key is not None else self.preset.api_key(),
                timeout=CLAIM_TIMEOUT_SECONDS,
                max_retries=CLAIM_MAX_RETRIES,
            )

    def usage_totals(self) -> TokenUsage:
        return self._usage.model_copy()

    def read_claims(self, user_request: str, reply: str) -> list[str]:
        """The actions this reply says were already done. Cached answers are free."""
        text = reply or ""
        if not text.strip():
            return []

        key = cache_key(
            self.model,
            self.prompt_version,
            user_request,
            text,
            "",
            label=CACHE_LABEL,
        )
        saved = self.cache.get(key)
        if saved is not None:
            return only_known_actions(list(saved.get("claimed_actions") or []))

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": CLAIM_SYSTEM_PROMPT},
            {"role": "user", "content": build_claim_question(user_request, text)},
        ]

        answer_text = self._ask(messages)
        try:
            actions = read_claimed_actions(answer_text)
        except JudgeError:
            # One more try, with a plain reminder. Then give up.
            messages.append({"role": "assistant", "content": answer_text})
            messages.append({"role": "user", "content": RETRY_REMINDER})
            actions = read_claimed_actions(self._ask(messages))

        self.cache.set(key, {"claimed_actions": actions})
        return actions

    # ---------- talking to the provider ----------

    def _ask(self, messages: list[dict[str, Any]]) -> str:
        started = time.perf_counter()
        try:
            answer = self._create_waiting_out_a_rate_limit(messages)
        except openai.APIError as problem:
            raise JudgeError(self._plain_message(problem)) from problem
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError) as problem:
            raise JudgeError(self._plain_message(problem)) from problem

        usage = getattr(answer, "usage", None)
        self._usage.add(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            time_ms=round((time.perf_counter() - started) * 1000, 3),
        )

        choices = list(getattr(answer, "choices", None) or [])
        if not choices:
            raise JudgeError("The claim reader gave an empty answer.")
        return str(getattr(choices[0].message, "content", "") or "")

    def _create(self, messages: list[dict[str, Any]]) -> Any:
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=CLAIM_TEMPERATURE,
        )

    def _create_waiting_out_a_rate_limit(self, messages: list[dict[str, Any]]) -> Any:
        """On a 429, wait as long as the provider asked, then try once more."""
        try:
            return self._create(messages)
        except openai.RateLimitError as problem:
            wait = seconds_to_wait(problem)
            if wait is None:
                raise
            self._sleep(wait)
            return self._create(messages)

    def _plain_message(self, problem: Exception) -> str:
        label = "The claim reader (" + self.model + ")"
        if isinstance(problem, openai.RateLimitError):
            return label + " hit its rate limit."
        if isinstance(problem, (openai.APIConnectionError, openai.APITimeoutError)):
            return label + " did not answer."
        status = getattr(problem, "status_code", None)
        if status is not None:
            return label + " gave error " + str(status) + "."
        return label + " failed."
