"""Reads the broken agents in bug_zoo/.

One folder is one broken agent:

    bug_zoo/fake-success/about.yaml    what is broken, in plain words
    bug_zoo/fake-success/script.yaml   the fake brain script

about.yaml must name a real test case and the check that is expected to catch
the bug, with the name the check has in the code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from agentqa.project_paths import BUG_ZOO_DIR

ABOUT_FILE_NAME = "about.yaml"
SCRIPT_FILE_NAME = "script.yaml"
NEEDED_FIELDS = ("name", "bug", "based_on", "test_case", "expected_check")


class ZooFileError(Exception):
    """A bug zoo folder is missing something or is wrong."""


@dataclass(frozen=True)
class BrokenAgent:
    """One broken agent, with the bug it carries."""

    name: str
    bug: str
    based_on: str
    test_case: str
    expected_check: str
    script_file: Path
    folder: Path


def load_broken_agent(folder: str | Path) -> BrokenAgent:
    path = Path(folder)
    about_file = path / ABOUT_FILE_NAME
    script_file = path / SCRIPT_FILE_NAME

    if not about_file.is_file():
        raise ZooFileError("No " + ABOUT_FILE_NAME + " in " + str(path))
    if not script_file.is_file():
        raise ZooFileError("No " + SCRIPT_FILE_NAME + " in " + str(path))

    try:
        raw = yaml.safe_load(about_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as problem:
        raise ZooFileError("Bad YAML in " + str(about_file) + ": " + str(problem)) from problem
    if not isinstance(raw, dict):
        raise ZooFileError("Bad " + str(about_file) + ": the file must be a YAML mapping.")

    for needed in NEEDED_FIELDS:
        if not str(raw.get(needed, "")).strip():
            raise ZooFileError("Bad " + str(about_file) + ": " + needed + " is missing.")

    return BrokenAgent(
        name=str(raw["name"]).strip(),
        bug=str(raw["bug"]).strip(),
        based_on=str(raw["based_on"]).strip(),
        test_case=str(raw["test_case"]).strip(),
        expected_check=str(raw["expected_check"]).strip(),
        script_file=script_file,
        folder=path,
    )


def load_broken_agents(folder: str | Path | None = None, only: str | None = None) -> list[BrokenAgent]:
    """Every broken agent, in folder name order."""
    directory = Path(folder) if folder else BUG_ZOO_DIR
    if not directory.is_dir():
        raise ZooFileError("No bug zoo folder at " + str(directory))

    agents = [load_broken_agent(one) for one in sorted(directory.iterdir()) if one.is_dir()]
    if not agents:
        raise ZooFileError("No broken agents found in " + str(directory))

    seen: dict[str, str] = {}
    for agent in agents:
        if agent.name in seen:
            raise ZooFileError(
                "Two broken agents use the name " + agent.name + ": " + seen[agent.name]
                + " and " + str(agent.folder)
            )
        seen[agent.name] = str(agent.folder)

    if only:
        picked = [agent for agent in agents if agent.name == only]
        if not picked:
            raise ZooFileError(
                "No broken agent called " + only + ". Found: " + ", ".join(sorted(seen))
            )
        return picked
    return agents
