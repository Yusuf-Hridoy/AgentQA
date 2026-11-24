"""Reads the hand-labelled claim examples in judge_calibration/.

These examples are how we check the claim reader itself. `agentqa check-judge`
runs the AI reader on each one and counts how often it agrees with the human
label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from agentqa.claims.claim_reader import WRITE_ACTIONS
from agentqa.project_paths import JUDGE_CALIBRATION_DIR

CLAIM_EXAMPLES_FILE_NAME = "claim_examples.yaml"


class ClaimExampleFileError(Exception):
    """The claim example file is missing or wrong."""


@dataclass(frozen=True)
class ClaimExample:
    """One reply, with the claims a person read in it."""

    id: str
    user_asks: str
    reply: str
    expected: list[str] = field(default_factory=list)


def load_claim_examples(file_path: str | Path | None = None) -> list[ClaimExample]:
    path = Path(file_path) if file_path else JUDGE_CALIBRATION_DIR / CLAIM_EXAMPLES_FILE_NAME
    if not path.is_file():
        raise ClaimExampleFileError("No claim example file at " + str(path))

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as problem:
        raise ClaimExampleFileError("Bad YAML in " + path.name + ": " + str(problem)) from problem

    entries = raw.get("examples") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ClaimExampleFileError("Bad claim example file: " + path.name + ". It needs an examples list.")

    examples = []
    for number, entry in enumerate(entries, start=1):
        where = "example " + str(number) + " in " + path.name
        if not isinstance(entry, dict):
            raise ClaimExampleFileError("Bad " + where + ": each example must be a mapping.")
        for needed in ("user_asks", "reply"):
            if not str(entry.get(needed, "")).strip():
                raise ClaimExampleFileError("Bad " + where + ": " + needed + " is missing.")
        expected = entry.get("expected", [])
        if not isinstance(expected, list):
            raise ClaimExampleFileError("Bad " + where + ": expected must be a list.")
        for action in expected:
            if action not in WRITE_ACTIONS:
                raise ClaimExampleFileError(
                    "Bad " + where + ": " + str(action) + " is not one of: " + ", ".join(WRITE_ACTIONS)
                )
        examples.append(
            ClaimExample(
                id=str(entry.get("id") or ("example-" + str(number))),
                user_asks=str(entry["user_asks"]).strip(),
                reply=str(entry["reply"]).strip(),
                expected=[str(one) for one in expected],
            )
        )
    return examples
