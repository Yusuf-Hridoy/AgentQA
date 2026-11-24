"""Copies a run folder, or a scorecard folder, into docs/.

Order matters here, so it is fixed in code: scan first, copy second, rebuild
the index third. Nothing is copied while a key is still unaccounted for.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from agentqa.project_paths import DOCS_DIR
from agentqa.publish.key_scanner import KeyFound, scan_folder
from agentqa.publish.site_index import (
    REPORTS_FOLDER_NAME,
    SCORECARD_FOLDER_NAME,
    write_index,
)
from agentqa.scorecard.scorecard_writer import SCORECARD_HTML_FILE_NAME

# Never copied into a public folder, whatever a person asks for.
SKIPPED_NAMES = (".env", ".env.local")


class PublishError(Exception):
    """The folder cannot be published, and the message says why."""


class KeyFoundError(PublishError):
    """A file looks like it holds an API key. Nothing was copied."""

    def __init__(self, found: list[KeyFound]) -> None:
        self.found = found
        said = "\n  ".join(one.as_text() for one in found)
        super().__init__(
            "Nothing was copied. A key must never reach a public folder.\n  " + said
        )


def is_scorecard_folder(folder: str | Path) -> bool:
    return (Path(folder) / SCORECARD_HTML_FILE_NAME).is_file()


def _keep(path: Path) -> bool:
    return path.name not in SKIPPED_NAMES


def _copy_tree(source: Path, target: Path) -> list[Path]:
    """Copy the folder, replacing what was published under that name before."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(source.rglob("*")):
        if not _keep(path):
            continue
        where = target / path.relative_to(source)
        if path.is_dir():
            where.mkdir(parents=True, exist_ok=True)
            continue
        where.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, where)
        copied.append(where)
    return copied


def publish_folder(
    source_folder: str | Path,
    name: str | None = None,
    docs_dir: str | Path | None = None,
) -> tuple[Path, list[Path], Path]:
    """Scan, copy, rebuild the index.

    Gives back where it landed, what was copied and the index that was written.
    """
    source = Path(source_folder)
    if not source.is_dir():
        raise PublishError("No folder at " + str(source))

    found = scan_folder(source)
    if found:
        raise KeyFoundError(found)

    docs = Path(docs_dir) if docs_dir else DOCS_DIR
    if is_scorecard_folder(source):
        target = docs / SCORECARD_FOLDER_NAME
    else:
        target = docs / REPORTS_FOLDER_NAME / (name or source.name)

    copied = _copy_tree(source, target)
    index = write_index(docs)
    return target, copied, index
