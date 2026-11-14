"""Who may use which tool.

employee: search_help_docs, create_ticket, and get_ticket or update_ticket on
          their own tickets only.
it_agent: every tool, on every ticket.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ALL_TOOL_NAMES = (
    "search_help_docs",
    "get_ticket",
    "create_ticket",
    "update_ticket",
    "assign_ticket",
    "send_notification",
)

ROLE_TOOLS: dict[str, set[str]] = {
    "employee": {"search_help_docs", "get_ticket", "create_ticket", "update_ticket"},
    "it_agent": set(ALL_TOOL_NAMES),
}

# For an employee these tools only work on tickets they own.
OWN_TICKET_ONLY_TOOLS = {"get_ticket", "update_ticket"}


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    reason: str = ""


def tools_for_role(role: str) -> set[str]:
    return set(ROLE_TOOLS.get(role, set()))


def check_permission(store: Any, acting_user: str, tool_name: str, inputs: dict[str, Any]) -> PermissionDecision:
    """Say whether this user may make this call right now."""
    user = store.users.get(acting_user)
    if user is None:
        return PermissionDecision(False, "Not allowed: unknown user " + str(acting_user) + ".")

    role = str(user["role"])
    allowed_tools = tools_for_role(role)
    if tool_name not in allowed_tools:
        return PermissionDecision(
            False,
            "Not allowed: a user with the role " + role + " cannot use " + tool_name + ".",
        )

    if role == "employee" and tool_name in OWN_TICKET_ONLY_TOOLS:
        ticket_id = inputs.get("ticket_id")
        ticket = store.ticket_or_none(ticket_id) if ticket_id else None
        # An unknown ticket is not a permission problem. The tool reports it.
        if ticket is not None and ticket.get("owner") != acting_user:
            return PermissionDecision(
                False,
                "Not allowed: ticket " + str(ticket_id) + " belongs to another user.",
            )

    return PermissionDecision(True)
