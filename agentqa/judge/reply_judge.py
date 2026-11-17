"""The AI judge: does the reply mean the right thing?

Word matching cannot tell "I made a ticket for your VPN problem" from "I could
not make a ticket". The judge can. It reads the user request, the agent reply
and one plain-English expectation, and answers pass or fail with one short
reason.

What the judge does not do: it never decides whether a forbidden word leaked,
and it never decides whether a tool was called. Those are facts, so plain code
checks them.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import openai

from agentqa.brains.brain_interface import TokenUsage
from agentqa.brains.brain_presets import BRAIN_GEMINI, preset_for
from agentqa.brains.rate_limit_wait import seconds_to_wait
from agentqa.judge.judge_cache import JudgeCache, cache_key
from agentqa.judge.judge_prompt import (
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM_PROMPT,
    RETRY_REMINDER,
    build_judge_question,
)

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
JUDGE_TIMEOUT_SECONDS = 60
JUDGE_MAX_RETRIES = 3
JUDGE_TEMPERATURE = 0.0

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)


class JudgeError(Exception):
    """The judge could not give an answer. The test becomes ERROR, not FAIL."""


@dataclass
class JudgeVerdict:
    """One judged reply."""

    passed: bool
    reason: str
    from_cache: bool = False


def read_verdict(text: str) -> JudgeVerdict:
    """Pull the verdict out of the judge's answer. Raises when it is not there."""
    cleaned = _FENCE.sub("", str(text or "")).strip()
    found = _JSON_OBJECT.search(cleaned)
    if not found:
        raise JudgeError("The judge did not answer with JSON.")
    try:
        answer = json.loads(found.group(0))
    except ValueError as problem:
        raise JudgeError("The judge answered with broken JSON.") from problem
    if not isinstance(answer, dict):
        raise JudgeError("The judge did not answer with a JSON object.")

    verdict = str(answer.get("verdict", "")).strip().lower()
    if verdict not in (VERDICT_PASS, VERDICT_FAIL):
        raise JudgeError("The judge gave no pass or fail verdict.")

    reason = str(answer.get("reason", "")).strip() or "No reason given."
    return JudgeVerdict(passed=verdict == VERDICT_PASS, reason=reason)


class ReplyJudge:
    """Asks a model whether one reply means the right thing."""

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
        self.prompt_version = JUDGE_PROMPT_VERSION
        self.cache = cache if cache is not None else JudgeCache(enabled=use_cache)
        self._usage = TokenUsage()

        if client is not None:
            self.client = client
        else:
            self.client = openai.OpenAI(
                base_url=self.preset.base(),
                api_key=api_key if api_key is not None else self.preset.api_key(),
                timeout=JUDGE_TIMEOUT_SECONDS,
                max_retries=JUDGE_MAX_RETRIES,
            )

    def usage_totals(self) -> TokenUsage:
        return self._usage.model_copy()

    def judge(self, user_request: str, reply: str | None, expectation: str) -> JudgeVerdict:
        """Pass or fail, with one short reason. Cached answers cost nothing."""
        text = reply or ""
        key = cache_key(self.model, self.prompt_version, user_request, text, expectation)

        saved = self.cache.get(key)
        if saved is not None:
            return JudgeVerdict(
                passed=bool(saved.get("passed")),
                reason=str(saved.get("reason", "")),
                from_cache=True,
            )

        question = build_judge_question(user_request, text, expectation)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        answer_text = self._ask(messages)
        try:
            verdict = read_verdict(answer_text)
        except JudgeError:
            # One more try, with a plain reminder. Then give up.
            messages.append({"role": "assistant", "content": answer_text})
            messages.append({"role": "user", "content": RETRY_REMINDER})
            verdict = read_verdict(self._ask(messages))

        self.cache.set(key, {"passed": verdict.passed, "reason": verdict.reason})
        return verdict

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
            raise JudgeError("The judge gave an empty answer.")
        return str(getattr(choices[0].message, "content", "") or "")

    def _create(self, messages: list[dict[str, Any]]) -> Any:
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=JUDGE_TEMPERATURE,
        )

    def _create_waiting_out_a_rate_limit(self, messages: list[dict[str, Any]]) -> Any:
        """On a 429, wait as long as the provider asked, then try once more.

        The judge shares a free tier with the agent, so it meets the same
        limits. Waiting beats losing the test.
        """
        try:
            return self._create(messages)
        except openai.RateLimitError as problem:
            wait = seconds_to_wait(problem)
            if wait is None:
                raise
            self._sleep(wait)
            return self._create(messages)

    def _plain_message(self, problem: Exception) -> str:
        label = "The judge (" + self.model + ")"
        if isinstance(problem, openai.RateLimitError):
            return label + " hit its rate limit."
        if isinstance(problem, (openai.APIConnectionError, openai.APITimeoutError)):
            return label + " did not answer."
        status = getattr(problem, "status_code", None)
        if status is not None:
            return label + " gave error " + str(status) + "."
        return label + " failed."
