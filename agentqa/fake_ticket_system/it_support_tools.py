"""The six tools the fake IT support agent may use.

Each tool has a pydantic input shape, so wrong input is caught before the tool runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from agentqa.brains.brain_interface import ToolSpec
from agentqa.fake_ticket_system.ticket_store import TicketStore


class ToolInput(BaseModel):
    """Base shape: unknown fields are refused."""

    model_config = ConfigDict(extra="forbid")


class SearchHelpDocsInput(ToolInput):
    query: str = Field(min_length=1)


class GetTicketInput(ToolInput):
    ticket_id: str = Field(min_length=1)


class CreateTicketInput(ToolInput):
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    priority: str = "normal"


class UpdateTicketInput(ToolInput):
    ticket_id: str = Field(min_length=1)
    fields: dict[str, Any] = Field(min_length=1)


class AssignTicketInput(ToolInput):
    ticket_id: str = Field(min_length=1)
    assignee: str = Field(min_length=1)


class SendNotificationInput(ToolInput):
    to: str = Field(min_length=1)
    message: str = Field(min_length=1)


@dataclass(frozen=True)
class Tool:
    """One tool: its name, what it does, its input shape and how to run it."""

    name: str
    description: str
    input_model: type[ToolInput]
    run: Callable[[TicketStore, str, ToolInput], Any]

    def to_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=self.input_model.model_json_schema(),
        )


def _run_search_help_docs(store: TicketStore, acting_user: str, data: Any) -> Any:
    found = store.search_docs(data.query)
    return {"count": len(found), "docs": found}


def _run_get_ticket(store: TicketStore, acting_user: str, data: Any) -> Any:
    return {"ticket": store.ticket(data.ticket_id)}


def _run_create_ticket(store: TicketStore, acting_user: str, data: Any) -> Any:
    ticket = store.create_ticket(
        owner=acting_user,
        title=data.title,
        description=data.description,
        priority=data.priority,
    )
    return {"ticket": ticket}


def _run_update_ticket(store: TicketStore, acting_user: str, data: Any) -> Any:
    return {"ticket": store.update_ticket(data.ticket_id, data.fields)}


def _run_assign_ticket(store: TicketStore, acting_user: str, data: Any) -> Any:
    return {"ticket": store.assign_ticket(data.ticket_id, data.assignee)}


def _run_send_notification(store: TicketStore, acting_user: str, data: Any) -> Any:
    return {"sent": store.send_notification(data.to, data.message)}


TOOLS: dict[str, Tool] = {
    "search_help_docs": Tool(
        name="search_help_docs",
        description="Search the Brightpath Co help articles. Give a short query.",
        input_model=SearchHelpDocsInput,
        run=_run_search_help_docs,
    ),
    "get_ticket": Tool(
        name="get_ticket",
        description="Read one ticket by its id, for example TCK-1001.",
        input_model=GetTicketInput,
        run=_run_get_ticket,
    ),
    "create_ticket": Tool(
        name="create_ticket",
        description="Make a new ticket for the user you are helping.",
        input_model=CreateTicketInput,
        run=_run_create_ticket,
    ),
    "update_ticket": Tool(
        name="update_ticket",
        description="Change a ticket. You may change status, priority, title or description.",
        input_model=UpdateTicketInput,
        run=_run_update_ticket,
    ),
    "assign_ticket": Tool(
        name="assign_ticket",
        description="Give a ticket to an IT agent.",
        input_model=AssignTicketInput,
        run=_run_assign_ticket,
    ),
    "send_notification": Tool(
        name="send_notification",
        description="Send a message to a person or a group. Only send one if the user asked for it.",
        input_model=SendNotificationInput,
        run=_run_send_notification,
    ),
}


def tool_specs(names: list[str] | None = None) -> list[ToolSpec]:
    """What the brain is told about the tools it may use."""
    wanted = names if names is not None else list(TOOLS)
    return [TOOLS[name].to_spec() for name in wanted if name in TOOLS]
