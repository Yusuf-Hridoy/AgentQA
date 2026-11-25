"""No test file may import another one through a `tests.` package path.

`tests/` has no `__init__.py`, so it is not a package. pytest puts the `tests`
folder itself on `sys.path`, not the folder above it, so one test file reaches
another by its plain name: `from conftest import ...`, `from test_scorecard
import ...`.

`from tests.test_scorecard import ...` only works when the repo root happens to
be on `sys.path` as well, which is what `python -m pytest` does and the plain
`pytest` command does not. So it passes on a developer's machine and fails in
CI with `ModuleNotFoundError: No module named 'tests'`, and the whole run stops
at collection with exit code 2 instead of running.

This test reads the import lines, so it catches the mistake however pytest was
started.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_FOLDER = Path(__file__).resolve().parent

# The name of this folder, which is exactly what a test file must not import.
FOLDER_NAME = TESTS_FOLDER.name

HOW_TO_FIX = (
    "Import it by its plain name instead, as every other test file does: "
    "`from test_scorecard import ...`, not `from "
    + FOLDER_NAME
    + ".test_scorecard import ...`."
)


def python_files() -> list[Path]:
    return sorted(TESTS_FOLDER.glob("*.py"))


def imported_names(file: Path) -> list[str]:
    """Every name this file imports, as written in the source."""
    # utf-8-sig, so a file an editor saved with a BOM still reads.
    tree = ast.parse(file.read_text(encoding="utf-8-sig"), filename=str(file))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def reaches_through_the_folder(name: str) -> bool:
    return name == FOLDER_NAME or name.startswith(FOLDER_NAME + ".")


def test_the_tests_folder_is_not_a_package():
    """If it ever becomes one, this whole rule changes and so must this test."""
    assert not (TESTS_FOLDER / "__init__.py").exists()


def test_there_are_test_files_to_check():
    """A glob that finds nothing would make the real test pass for free."""
    assert len(python_files()) > 10


def test_no_test_file_imports_another_through_the_tests_package():
    bad = []
    for file in python_files():
        for name in imported_names(file):
            if reaches_through_the_folder(name):
                bad.append(file.name + " imports " + name)
    assert not bad, "These imports break the plain `pytest` command. " + HOW_TO_FIX + " " + str(bad)


def test_the_rule_catches_the_import_that_broke_ci(tmp_path):
    """The exact line that failed in CI, to prove the check sees it."""
    broken = tmp_path / "test_broken.py"
    broken.write_text(
        "from " + FOLDER_NAME + ".test_scorecard import a_run_folder\n",
        encoding="utf-8",
    )
    assert any(reaches_through_the_folder(name) for name in imported_names(broken))


def test_the_rule_leaves_the_right_import_alone(tmp_path):
    good = tmp_path / "test_good.py"
    good.write_text("from test_scorecard import a_run_folder\n", encoding="utf-8")
    assert not any(reaches_through_the_folder(name) for name in imported_names(good))
