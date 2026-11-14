"""Reads and checks the YAML files in test_cases/."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from agentqa.project_paths import TEST_CASES_DIR
from agentqa.test_case_schema import TestCase


class TestCaseFileError(Exception):
    """A test case file is missing, unreadable or wrong."""


def _readable_validation_error(file_path: Path, error: ValidationError) -> str:
    lines = ["Bad test case file: " + file_path.name]
    for problem in error.errors():
        where = ".".join(str(piece) for piece in problem["loc"]) or "file"
        lines.append("  " + where + ": " + problem["msg"])
    return "\n".join(lines)


def load_test_case_file(file_path: str | Path) -> TestCase:
    """Read one test case file and check its shape."""
    path = Path(file_path)
    if not path.is_file():
        raise TestCaseFileError("No test case file at " + str(path))

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as problem:
        raise TestCaseFileError("Bad YAML in " + path.name + ": " + str(problem)) from problem

    if not isinstance(raw, dict):
        raise TestCaseFileError("Bad test case file: " + path.name + ". The file must be a YAML mapping.")

    try:
        test_case = TestCase(**raw)
    except ValidationError as problem:
        raise TestCaseFileError(_readable_validation_error(path, problem)) from problem

    test_case.source_file = path.name
    return test_case


def load_test_cases(folder: str | Path | None = None, only: str | None = None) -> list[TestCase]:
    """Read every test case file in a folder, in file name order."""
    directory = Path(folder) if folder else TEST_CASES_DIR
    if not directory.is_dir():
        raise TestCaseFileError("No test case folder at " + str(directory))

    files = sorted(list(directory.glob("*.yaml")) + list(directory.glob("*.yml")))
    if not files:
        raise TestCaseFileError("No test case files found in " + str(directory))

    cases = [load_test_case_file(path) for path in files]

    seen: dict[str, str] = {}
    for case in cases:
        if case.id in seen:
            raise TestCaseFileError(
                "Two test cases use the id " + case.id + ": " + seen[case.id] + " and " + str(case.source_file)
            )
        seen[case.id] = str(case.source_file)

    if only:
        picked = [case for case in cases if case.id == only]
        if not picked:
            raise TestCaseFileError(
                "No test case with the id " + only + ". Found: " + ", ".join(sorted(seen))
            )
        return picked

    return cases
