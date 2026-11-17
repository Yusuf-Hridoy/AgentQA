"""Reads the hand-labelled judge examples in judge_calibration/.

These examples are how we check the judge itself. `agentqa check-judge` runs
the judge on each one and counts how often it agrees with the human label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from agentqa.project_paths import JUDGE_CALIBRATION_DIR

JUDGE_EXAMPLES_FILE_NAME = "judge_examples.yaml"
LABEL_PASS = "pass"
LABEL_FAIL = "fail"


class JudgeExampleFileError(Exception):
    """The judge example file is missing or wrong."""


@dataclass(frozen=True)
class JudgeExample:
    """One reply a person already judged."""

    id: str
    user_asks: str
    reply: str
    expectation: str
    label: str

    @property
    def should_pass(self) -> bool:
        return self.label == LABEL_PASS


def load_judge_examples(file_path: str | Path | None = None) -> list[JudgeExample]:
    path = Path(file_path) if file_path else JUDGE_CALIBRATION_DIR / JUDGE_EXAMPLES_FILE_NAME
    if not path.is_file():
        raise JudgeExampleFileError("No judge example file at " + str(path))

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as problem:
        raise JudgeExampleFileError("Bad YAML in " + path.name + ": " + str(problem)) from problem

    entries = raw.get("examples") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise JudgeExampleFileError("Bad judge example file: " + path.name + ". It needs an examples list.")

    examples = []
    for number, entry in enumerate(entries, start=1):
        where = "example " + str(number) + " in " + path.name
        if not isinstance(entry, dict):
            raise JudgeExampleFileError("Bad " + where + ": each example must be a mapping.")
        label = str(entry.get("label", "")).strip().lower()
        if label not in (LABEL_PASS, LABEL_FAIL):
            raise JudgeExampleFileError("Bad " + where + ": label must be pass or fail.")
        for needed in ("user_asks", "reply", "expectation"):
            if not str(entry.get(needed, "")).strip():
                raise JudgeExampleFileError("Bad " + where + ": " + needed + " is missing.")
        examples.append(
            JudgeExample(
                id=str(entry.get("id") or ("example-" + str(number))),
                user_asks=str(entry["user_asks"]).strip(),
                reply=str(entry["reply"]).strip(),
                expectation=str(entry["expectation"]).strip(),
                label=label,
            )
        )
    return examples
