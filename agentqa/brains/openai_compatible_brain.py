"""A brain that asks a real model what to do next.

One file for three providers, because local Ollama, Groq and Gemini all speak
the OpenAI chat completions API. Only the base URL, the key and the model id
differ, and those live in brain_presets.py.

What this file is careful about:

* One model answer can hold several tool calls. They go in a queue and come out
  one per step, so none is dropped.
* Broken JSON in the tool inputs is passed on as {"_raw": "..."} instead of
  being hidden. The gateway then refuses it and the trace shows the mistake.
* <think> blocks are cut out of the final reply.
* An empty answer is reported as empty, never replaced with words of ours.
  The reply checks fail it, and the judge is never shown it.
* Tokens and time are counted for every call.
* After the SDK has retried, a failure becomes a BrainError, which marks one
  test ERROR. It never stops the run.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import openai

from agentqa.brains.brain_errors import BrainError
from agentqa.brains.brain_interface import (
    BrainAction,
    FinalReply,
    Message,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from agentqa.brains.brain_presets import BrainPreset
from agentqa.brains.rate_limit_wait import seconds_to_wait

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_RETRIES = 3
TEMPERATURE = 0.0

RAW_INPUT_KEY = "_raw"

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_CLOSE_THINK = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)

# JSON schema keywords that some providers refuse. The gateway checks the input
# anyway, so dropping them loses nothing.
_SCHEMA_KEYWORDS_TO_DROP = ("title", "additionalProperties", "$schema", "default")

# Inside these, the keys are names the agent chose, not schema keywords. A tool
# with a field called "title" must keep it, or the schema asks for a field that
# is not there.
_SCHEMA_NAME_MAPS = ("properties", "patternProperties", "$defs", "definitions")


def strip_thinking(text: str) -> str:
    """Take out <think>...</think>, so the reply is what the user would see."""
    if not text:
        return ""
    cleaned = _THINK_BLOCK.sub("", text)
    if "</think>" in cleaned.lower():
        cleaned = _CLOSE_THINK.sub("", cleaned)
    if "<think>" in cleaned.lower():
        cleaned = _OPEN_THINK.sub("", cleaned)
    return cleaned.strip()


def clean_schema(schema: Any) -> Any:
    """A plain JSON schema that every provider accepts.

    Only schema keywords are dropped. Field names are never touched, so a tool
    with a field called "title" or "default" keeps it.
    """
    if isinstance(schema, dict):
        cleaned: dict[str, Any] = {}
        for key, value in schema.items():
            if key in _SCHEMA_KEYWORDS_TO_DROP:
                continue
            if key in _SCHEMA_NAME_MAPS and isinstance(value, dict):
                cleaned[key] = {name: clean_schema(sub) for name, sub in value.items()}
            else:
                cleaned[key] = clean_schema(value)
        return cleaned
    if isinstance(schema, list):
        return [clean_schema(item) for item in schema]
    return schema


def tools_as_openai(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Our ToolSpec list in the format the chat completions API wants."""
    wanted = []
    for tool in tools:
        parameters = clean_schema(tool.input_schema) or {"type": "object", "properties": {}}
        wanted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                },
            }
        )
    return wanted


def one_tool_call(line: Message) -> dict[str, Any]:
    """One assistant tool call, with the provider's own extra put back."""
    call: dict[str, Any] = {
        "id": line.tool_call_id or "call_unknown",
        "type": "function",
        "function": {
            "name": line.tool,
            "arguments": json.dumps(line.tool_inputs or {}, default=str),
        },
    }
    if line.tool_extra:
        call["extra_content"] = line.tool_extra
    return call


def conversation_as_openai(conversation: list[Message]) -> list[dict[str, Any]]:
    """Our conversation in the format the chat completions API wants."""
    messages: list[dict[str, Any]] = []
    for line in conversation:
        if line.role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": line.tool_call_id or "call_unknown",
                    "content": line.content,
                }
            )
        elif line.role == "assistant" and line.tool:
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [one_tool_call(line)],
                }
            )
        else:
            messages.append({"role": line.role, "content": line.content})
    return messages


def _reason_in(body: Any) -> str:
    """Dig the human sentence out of an error body, whatever shape it has."""
    if isinstance(body, list):
        for item in body:
            found = _reason_in(item)
            if found:
                return found
        return ""
    if isinstance(body, dict):
        for key in ("message", "error", "detail"):
            if key in body:
                found = _reason_in(body[key])
                if found:
                    return found
        return ""
    return str(body) if body else ""


def provider_extra_of(raw_call: Any) -> dict[str, Any] | None:
    """Whatever the provider hung on its own tool call.

    Gemini 3 puts a thought_signature here and refuses the next request if it
    does not come back. We never read it, we only carry it.
    """
    extra = getattr(raw_call, "extra_content", None)
    if extra is None:
        return None
    if hasattr(extra, "model_dump"):
        extra = extra.model_dump()
    return dict(extra) if isinstance(extra, dict) else None


def tool_call_from_model(raw_call: Any) -> ToolCall:
    """One tool call from the model. Bad JSON is kept, not hidden."""
    function = raw_call.function
    arguments = getattr(function, "arguments", None) or ""
    try:
        inputs = json.loads(arguments) if str(arguments).strip() else {}
    except (ValueError, TypeError):
        inputs = {RAW_INPUT_KEY: str(arguments)}
    if not isinstance(inputs, dict):
        inputs = {RAW_INPUT_KEY: str(arguments)}
    return ToolCall(
        tool=str(getattr(function, "name", "") or ""),
        inputs=inputs,
        call_id=str(getattr(raw_call, "id", "") or "") or None,
        provider_extra=provider_extra_of(raw_call),
    )


class OpenAICompatibleBrain:
    """Asks an OpenAI-compatible model for the next action."""

    def __init__(
        self,
        preset: BrainPreset,
        model: str | None = None,
        api_key: str | None = None,
        client: Any = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        sleeper: Any = None,
    ) -> None:
        self.preset = preset
        # Swapped out in the unit tests, so no test ever really waits.
        self._sleep = sleeper if sleeper is not None else time.sleep
        self.name = preset.name
        self.model = model or preset.model()
        self.max_retries = max_retries
        self._secret: str | None = None
        self._usage = TokenUsage()
        self._waiting_tool_calls: list[ToolCall] = []
        self._send_parallel_flag = preset.send_parallel_tool_calls
        self._extra_body = dict(preset.extra_body)
        self._reasoning_effort = preset.reasoning_effort

        if client is not None:
            self.client = client
        else:
            key = api_key if api_key is not None else preset.api_key()
            self._secret = key
            self.client = openai.OpenAI(
                base_url=preset.base(),
                api_key=key,
                timeout=timeout,
                max_retries=max_retries,
            )

    # ---------- what the agent loop uses ----------

    def usage_totals(self) -> TokenUsage:
        """Tokens and time used so far. A copy, so a caller cannot change it."""
        return self._usage.model_copy()

    def next_action(self, conversation: list[Message], tools: list[ToolSpec]) -> BrainAction:
        """The next thing the model wants to do."""
        if self._waiting_tool_calls:
            return self._waiting_tool_calls.pop(0)

        answer = self._ask(conversation_as_openai(conversation), tools_as_openai(tools))
        choices = list(getattr(answer, "choices", None) or [])
        if not choices:
            return FinalReply(reply="", finish_reason=None, empty_reply=True)
        choice = choices[0]
        finish_reason = str(getattr(choice, "finish_reason", "") or "") or None

        raw_calls = list(getattr(choice.message, "tool_calls", None) or [])
        if raw_calls:
            calls = [tool_call_from_model(one) for one in raw_calls]
            self._waiting_tool_calls = calls[1:]
            return calls[0]

        reply = strip_thinking(str(getattr(choice.message, "content", "") or ""))
        return FinalReply(reply=reply, finish_reason=finish_reason, empty_reply=not reply)

    # ---------- talking to the provider ----------

    def _request_fields(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": TEMPERATURE,
        }
        if tools:
            fields["tools"] = tools
            fields["tool_choice"] = "auto"
            if self._send_parallel_flag:
                fields["parallel_tool_calls"] = False
        if self._extra_body:
            fields["extra_body"] = dict(self._extra_body)
        if self._reasoning_effort:
            fields["reasoning_effort"] = self._reasoning_effort
        return fields

    def _ask(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        started = time.perf_counter()
        try:
            answer = self._create_waiting_out_a_rate_limit(messages, tools)
        except openai.APIError as problem:
            raise BrainError(self._plain_message(problem)) from problem
        except (ValueError, TypeError, ConnectionError, TimeoutError, OSError) as problem:
            raise BrainError(self._plain_message(problem)) from problem

        self._count(answer, (time.perf_counter() - started) * 1000)
        return answer

    def _create_waiting_out_a_rate_limit(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Any:
        """On a 429, wait as long as the provider asked, then try once more.

        The SDK has already backed off and retried by now. A free tier often
        says "retry in 18 seconds", and waiting those seconds is far better
        than losing the test.
        """
        try:
            return self._create(messages, tools)
        except openai.RateLimitError as problem:
            wait = seconds_to_wait(problem)
            if wait is None:
                raise
            self._sleep(wait)
            return self._create(messages, tools)

    def _create(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        """One request. Optional extras are dropped once if the provider says no."""
        try:
            return self.client.chat.completions.create(**self._request_fields(messages, tools))
        except openai.BadRequestError:
            if not (self._send_parallel_flag or self._extra_body or self._reasoning_effort):
                raise
            # Some providers refuse parallel_tool_calls or the thinking switch.
            # Drop them for the rest of the run and ask once more.
            self._send_parallel_flag = False
            self._extra_body = {}
            self._reasoning_effort = None
            return self.client.chat.completions.create(**self._request_fields(messages, tools))

    def _count(self, answer: Any, time_ms: float) -> None:
        usage = getattr(answer, "usage", None)
        self._usage.add(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            time_ms=round(time_ms, 3),
        )

    # ---------- short, plain, key-free error text ----------

    def _hide_key(self, text: str) -> str:
        if self._secret and self._secret in text:
            return text.replace(self._secret, "***")
        return text

    def _plain_message(self, problem: Exception) -> str:
        label = self.preset.label
        tries = str(self.max_retries)
        if isinstance(problem, openai.RateLimitError):
            return label + " hit its rate limit after " + tries + " tries."
        if isinstance(problem, (openai.APIConnectionError, openai.APITimeoutError)):
            return label + " did not answer after " + tries + " tries."
        status = getattr(problem, "status_code", None)
        if status is not None:
            # A 4xx usually means AgentQA asked wrongly, so say what the
            # provider objected to. Without it a whole run is undebuggable.
            return label + " gave error " + str(status) + ": " + self._short_detail(problem)
        return label + " failed: " + self._short_detail(problem)

    def _short_detail(self, problem: Exception) -> str:
        """What the provider actually complained about, with no key in it.

        The SDK's own message can be as useless as "Error code: 400", so the
        body is read first. A 4xx is usually AgentQA's mistake, and a run is
        not debuggable without the reason.
        """
        text = _reason_in(getattr(problem, "body", None))
        if not text:
            text = getattr(problem, "message", None) or str(problem)
        flat = " ".join(self._hide_key(str(text or "")).split())
        return flat[:220] if flat else problem.__class__.__name__
