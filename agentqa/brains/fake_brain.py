"""A brain that does not think.

It plays back the actions listed in fake_brain_scripts/<test id>.yaml, one per step.
It ignores the conversation, so a run is free, fast and the same every time.

A script file looks like this:

    actions:
      - tool: search_help_docs
        inputs: { query: "printer not printing" }
      - reply: "I made a ticket for your printer problem."
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agentqa.brains.brain_interface import (
    BrainAction,
    FinalReply,
    Message,
    TokenUsage,
    ToolCall,
    ToolSpec,
)
from agentqa.project_paths import FAKE_BRAIN_SCRIPTS_DIR

NO_MORE_ACTIONS_REPLY = "The fake brain script has no more actions."


class FakeBrainScriptError(Exception):
    """A fake brain script is missing or wrong."""


def _action_from_entry(entry: Any, file_name: str, number: int) -> BrainAction:
    where = "action " + str(number) + " in " + file_name
    if not isinstance(entry, dict):
        raise FakeBrainScriptError("Bad " + where + ": each action must be a mapping.")

    if "reply" in entry:
        reply = entry["reply"]
        if not isinstance(reply, str) or not reply.strip():
            raise FakeBrainScriptError("Bad " + where + ": reply must be text.")
        return FinalReply(reply=reply)

    if "tool" in entry:
        tool = entry["tool"]
        inputs = entry.get("inputs", {})
        if not isinstance(tool, str) or not tool.strip():
            raise FakeBrainScriptError("Bad " + where + ": tool must be a tool name.")
        if not isinstance(inputs, dict):
            raise FakeBrainScriptError("Bad " + where + ": inputs must be a mapping.")
        return ToolCall(tool=tool, inputs=inputs)

    raise FakeBrainScriptError("Bad " + where + ": an action needs either tool or reply.")


class FakeBrain:
    """Plays back a fixed list of actions."""

    name = "fake"

    def __init__(self, actions: list[BrainAction], source: str = "memory") -> None:
        self.actions = list(actions)
        self.source = source
        self._next_index = 0

    @classmethod
    def from_script_file(cls, file_path: str | Path) -> "FakeBrain":
        path = Path(file_path)
        if not path.is_file():
            raise FakeBrainScriptError("No fake brain script at " + str(path))
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as problem:
            raise FakeBrainScriptError("Bad YAML in " + path.name + ": " + str(problem)) from problem

        if not isinstance(raw, dict) or "actions" not in raw:
            raise FakeBrainScriptError("Bad fake brain script: " + path.name + ". It needs an actions list.")
        entries = raw["actions"]
        if not isinstance(entries, list) or not entries:
            raise FakeBrainScriptError("Bad fake brain script: " + path.name + ". actions must be a list.")

        actions = [_action_from_entry(entry, path.name, number) for number, entry in enumerate(entries, start=1)]
        return cls(actions, source=path.name)

    @classmethod
    def for_test_case(cls, test_id: str, folder: str | Path | None = None) -> "FakeBrain":
        directory = Path(folder) if folder else FAKE_BRAIN_SCRIPTS_DIR
        return cls.from_script_file(directory / (test_id + ".yaml"))

    def usage_totals(self) -> TokenUsage:
        """The fake brain costs nothing, so every number is zero."""
        return TokenUsage()

    def next_action(self, conversation: list[Message], tools: list[ToolSpec]) -> BrainAction:
        """Give the next scripted action. The conversation is ignored on purpose."""
        if self._next_index >= len(self.actions):
            return FinalReply(reply=NO_MORE_ACTIONS_REPLY)
        action = self.actions[self._next_index]
        self._next_index += 1
        return action
