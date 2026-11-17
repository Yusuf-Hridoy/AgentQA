"""The shape of a test case file in test_cases/.

Every file is checked against this shape, so a typo is caught before a test runs.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentqa.fake_ticket_system.tool_permissions import ALL_TOOL_NAMES

TestCaseType = Literal["normal", "unclear", "permission", "hidden_attack", "broken_tool", "loop"]


def _check_tool_names(names: list[str]) -> list[str]:
    for name in names:
        if name not in ALL_TOOL_NAMES:
            raise ValueError(
                "Unknown tool: " + str(name) + ". Known tools: " + ", ".join(ALL_TOOL_NAMES)
            )
    return names


class TicketSystemAfter(BaseModel):
    """What must be true in the ticket store when the run is over."""

    model_config = ConfigDict(extra="forbid")

    new_tickets: int | None = None
    new_ticket_fields: dict[str, Any] = Field(default_factory=dict)
    changed_tickets: int | None = None
    ticket_fields: dict[str, dict[str, Any]] = Field(default_factory=dict)
    notifications_sent: int | None = None

    def is_empty(self) -> bool:
        return (
            self.new_tickets is None
            and not self.new_ticket_fields
            and self.changed_tickets is None
            and not self.ticket_fields
            and self.notifications_sent is None
        )


class TestCase(BaseModel):
    """One AgentQA test case."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: TestCaseType
    acting_user: str = Field(min_length=1)
    user_asks: str = Field(min_length=1)

    should_use: list[str] = Field(default_factory=list)
    must_never_use: list[str] = Field(default_factory=list)
    max_calls: dict[str, int] = Field(default_factory=dict)
    expect_blocked: list[str] = Field(default_factory=list)

    ticket_system_after: TicketSystemAfter = Field(default_factory=TicketSystemAfter)

    reply_should_say: list[str] = Field(default_factory=list)
    # One plain-English sentence for the AI judge. Used instead of
    # reply_should_say when the run has a judge.
    reply_should_mean: str | None = None
    reply_must_not_say: list[str] = Field(default_factory=list)
    expect_question: bool = False

    tool_failures: dict[str, int] = Field(default_factory=dict)
    max_steps: int = Field(default=8, ge=1, le=50)
    critical: bool = False

    # Filled in by the loader. Not written in the YAML file.
    source_file: str | None = None

    @field_validator("should_use", "must_never_use", "expect_blocked")
    @classmethod
    def _tool_lists_are_known(cls, value: list[str]) -> list[str]:
        return _check_tool_names(value)

    @field_validator("max_calls", "tool_failures")
    @classmethod
    def _tool_maps_are_known(cls, value: dict[str, int]) -> dict[str, int]:
        _check_tool_names(list(value))
        for tool, number in value.items():
            if number < 0:
                raise ValueError("The number for " + tool + " cannot be below zero.")
        return value
