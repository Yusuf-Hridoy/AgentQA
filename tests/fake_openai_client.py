"""A stand-in for the OpenAI client, so the tests never touch the network.

It gives back answers from a list, one per call, and remembers what it was
asked. A test can also make it raise, to check how AgentQA handles a failure.

Nothing here needs a key, a server or an internet connection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction
    type: str = "function"


@dataclass
class FakeMessage:
    content: str | None = None
    tool_calls: list[FakeToolCall] | None = None


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str = "stop"


@dataclass
class FakeUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class FakeAnswer:
    choices: list[FakeChoice]
    usage: FakeUsage = field(default_factory=FakeUsage)


def text_answer(
    text: str | None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    finish_reason: str = "stop",
) -> FakeAnswer:
    return FakeAnswer(
        choices=[FakeChoice(message=FakeMessage(content=text), finish_reason=finish_reason)],
        usage=FakeUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def tool_answer(
    calls: list[tuple[str, str, str]],
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> FakeAnswer:
    """calls is a list of (call id, tool name, arguments as text)."""
    tool_calls = [
        FakeToolCall(id=call_id, function=FakeFunction(name=name, arguments=arguments))
        for call_id, name, arguments in calls
    ]
    return FakeAnswer(
        choices=[FakeChoice(message=FakeMessage(content=None, tool_calls=tool_calls))],
        usage=FakeUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class FakeCompletions:
    def __init__(self, owner: "FakeOpenAIClient") -> None:
        self.owner = owner

    def create(self, **fields: Any) -> FakeAnswer:
        self.owner.requests.append(fields)
        if not self.owner.answers:
            raise AssertionError("The fake client ran out of answers.")
        answer = self.owner.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class FakeChat:
    def __init__(self, owner: "FakeOpenAIClient") -> None:
        self.completions = FakeCompletions(owner)


class FakeModels:
    def __init__(self, owner: "FakeOpenAIClient") -> None:
        self.owner = owner

    def list(self) -> Any:
        if isinstance(self.owner.model_list, Exception):
            raise self.owner.model_list
        return self.owner.model_list


@dataclass
class FakeModelRow:
    id: str


@dataclass
class FakeModelList:
    data: list[FakeModelRow]


def model_list(*names: str) -> FakeModelList:
    return FakeModelList(data=[FakeModelRow(id=name) for name in names])


class FakeOpenAIClient:
    """Answers come out in the order they were put in."""

    def __init__(self, answers: list[Any] | None = None, models: Any = None) -> None:
        self.answers = list(answers or [])
        self.model_list = models if models is not None else model_list()
        self.requests: list[dict[str, Any]] = []
        self.chat = FakeChat(self)
        self.models = FakeModels(self)

    @property
    def call_count(self) -> int:
        return len(self.requests)
