"""The fake ticket system data, held in memory.

Every test gets its own fresh copy, so one test can never change another one.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from agentqa.project_paths import FAKE_DATA_DIR

ALLOWED_STATUS = ("open", "in_progress", "resolved", "closed")
ALLOWED_PRIORITY = ("low", "normal", "high")

# Very common words are ignored when searching the help docs.
SEARCH_STOPWORDS = frozenset(
    {"the", "and", "for", "not", "can", "cannot", "with", "please", "work", "does",
     "from", "that", "this", "will", "you", "my", "me", "is", "are", "was", "it"}
)

# Fields a user may change with update_ticket.
UPDATABLE_FIELDS = ("status", "priority", "title", "description")

_FILE_CACHE: dict[str, list[dict[str, Any]]] = {}


class TicketStoreError(Exception):
    """A tool asked the store for something it cannot do."""


def _read_json(data_dir: Path, name: str) -> list[dict[str, Any]]:
    key = str(data_dir / name)
    if key not in _FILE_CACHE:
        with open(data_dir / name, "r", encoding="utf-8") as handle:
            _FILE_CACHE[key] = json.load(handle)
    return _FILE_CACHE[key]


class TicketStore:
    """Users, tickets, help docs and sent notifications for one test run."""

    def __init__(
        self,
        users: list[dict[str, Any]],
        tickets: list[dict[str, Any]],
        help_docs: list[dict[str, Any]],
    ) -> None:
        self.users: dict[str, dict[str, Any]] = {u["user_id"]: copy.deepcopy(u) for u in users}
        self.tickets: dict[str, dict[str, Any]] = {t["ticket_id"]: copy.deepcopy(t) for t in tickets}
        self.help_docs: list[dict[str, Any]] = copy.deepcopy(help_docs)
        self.notifications: list[dict[str, str]] = []

        self._tickets_at_start: dict[str, dict[str, Any]] = copy.deepcopy(self.tickets)
        self._next_ticket_number = self._first_free_ticket_number()

    @classmethod
    def load_fresh(cls, data_dir: Path | None = None) -> "TicketStore":
        """Build a brand new store from the files in fake_data."""
        folder = Path(data_dir) if data_dir else FAKE_DATA_DIR
        return cls(
            users=_read_json(folder, "users.json"),
            tickets=_read_json(folder, "tickets.json"),
            help_docs=_read_json(folder, "help_docs.json"),
        )

    # ---------- reading ----------

    def _first_free_ticket_number(self) -> int:
        numbers = []
        for ticket_id in self.tickets:
            tail = ticket_id.split("-")[-1]
            if tail.isdigit():
                numbers.append(int(tail))
        return (max(numbers) + 1) if numbers else 1001

    def role_of(self, user_id: str) -> str:
        user = self.users.get(user_id)
        if user is None:
            raise TicketStoreError("Unknown user: " + user_id)
        return str(user["role"])

    def ticket_or_none(self, ticket_id: str) -> dict[str, Any] | None:
        return self.tickets.get(ticket_id)

    def ticket(self, ticket_id: str) -> dict[str, Any]:
        found = self.tickets.get(ticket_id)
        if found is None:
            raise TicketStoreError("No ticket with id " + ticket_id)
        return found

    def search_docs(self, query: str) -> list[dict[str, Any]]:
        words = [w.strip(".,!?") for w in query.lower().split()]
        words = [w for w in words if len(w) > 2 and w not in SEARCH_STOPWORDS]
        hits = []
        for doc in self.help_docs:
            text = doc["title"] + " " + doc["body"] + " " + " ".join(doc["tags"])
            text = text.lower()
            score = sum(1 for word in words if word in text)
            if score:
                hits.append((score, doc))
        hits.sort(key=lambda pair: pair[0], reverse=True)
        return [copy.deepcopy(doc) for _, doc in hits[:3]]

    # ---------- writing ----------

    def create_ticket(self, owner: str, title: str, description: str, priority: str) -> dict[str, Any]:
        if priority not in ALLOWED_PRIORITY:
            raise TicketStoreError("Priority must be one of: " + ", ".join(ALLOWED_PRIORITY))
        ticket_id = "TCK-" + str(self._next_ticket_number)
        self._next_ticket_number += 1
        ticket = {
            "ticket_id": ticket_id,
            "owner": owner,
            "title": title,
            "description": description,
            "status": "open",
            "priority": priority,
            "assignee": None,
        }
        self.tickets[ticket_id] = ticket
        return copy.deepcopy(ticket)

    def update_ticket(self, ticket_id: str, fields: dict[str, Any]) -> dict[str, Any]:
        ticket = self.ticket(ticket_id)
        if not fields:
            raise TicketStoreError("No fields given to update.")
        for name, value in fields.items():
            if name not in UPDATABLE_FIELDS:
                raise TicketStoreError(
                    "Cannot change " + name + " with update_ticket. You may change: "
                    + ", ".join(UPDATABLE_FIELDS) + "."
                )
            if name == "status" and value not in ALLOWED_STATUS:
                raise TicketStoreError("Status must be one of: " + ", ".join(ALLOWED_STATUS))
            if name == "priority" and value not in ALLOWED_PRIORITY:
                raise TicketStoreError("Priority must be one of: " + ", ".join(ALLOWED_PRIORITY))
        ticket.update(fields)
        return copy.deepcopy(ticket)

    def assign_ticket(self, ticket_id: str, assignee: str) -> dict[str, Any]:
        ticket = self.ticket(ticket_id)
        if assignee not in self.users:
            raise TicketStoreError("Unknown user: " + assignee)
        ticket["assignee"] = assignee
        return copy.deepcopy(ticket)

    def send_notification(self, to: str, message: str) -> dict[str, str]:
        sent = {"to": to, "message": message}
        self.notifications.append(sent)
        return copy.deepcopy(sent)

    # ---------- what changed ----------

    def new_tickets(self) -> list[dict[str, Any]]:
        """Tickets that did not exist when the test started."""
        return [
            copy.deepcopy(ticket)
            for ticket_id, ticket in self.tickets.items()
            if ticket_id not in self._tickets_at_start
        ]

    def changed_ticket_ids(self) -> list[str]:
        """Tickets that existed at the start and are different now."""
        return [
            ticket_id
            for ticket_id, before in self._tickets_at_start.items()
            if self.tickets.get(ticket_id) != before
        ]

    def snapshot(self) -> dict[str, Any]:
        """A plain picture of the store, for the saved results."""
        return {
            "tickets": copy.deepcopy(list(self.tickets.values())),
            "notifications": copy.deepcopy(self.notifications),
            "new_ticket_ids": [t["ticket_id"] for t in self.new_tickets()],
            "changed_ticket_ids": self.changed_ticket_ids(),
        }
