"""Reads the run folders, checks they are fair to compare, writes the scorecard.

    reports/scorecard-<date>_<time>/
        scorecard.html
        scorecard.md
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from agentqa.project_paths import REPORTS_DIR
from agentqa.report.report_builder import ReportFolderError, load_results
from agentqa.scorecard.scorecard_fairness import UnfairComparisonError, check_fair, shared_facts
from agentqa.scorecard.scorecard_html import build_scorecard_html
from agentqa.scorecard.scorecard_markdown import scorecard_markdown, scorecard_table
from agentqa.scorecard.scorecard_rows import ScorecardRow, build_rows, folder_label

SCORECARD_HTML_FILE_NAME = "scorecard.html"
SCORECARD_MARKDOWN_FILE_NAME = "scorecard.md"


def scorecard_folder_name(when: datetime | None = None) -> str:
    moment = when or datetime.now()
    return "scorecard-" + moment.strftime("%Y-%m-%d_%H-%M-%S")


def make_scorecard_folder(reports_dir: str | Path | None = None, when: datetime | None = None) -> Path:
    base = Path(reports_dir) if reports_dir else REPORTS_DIR
    folder = base / scorecard_folder_name(when)
    counter = 2
    while folder.exists():
        folder = base / (scorecard_folder_name(when) + "-" + str(counter))
        counter += 1
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def load_runs(run_folders: list[str | Path]) -> list[tuple[str, dict[str, Any]]]:
    """Each run folder as (its name, its results.json)."""
    return [(folder_label(one), load_results(one)) for one in run_folders]


def build_scorecard(run_folders: list[str | Path]) -> tuple[list[ScorecardRow], dict[str, Any]]:
    """The rows and the shared facts. Raises when the runs are not comparable."""
    runs = load_runs(run_folders)
    check_fair(runs)
    return build_rows(runs), shared_facts(runs)


def save_scorecard(
    run_folders: list[str | Path],
    reports_dir: str | Path | None = None,
) -> tuple[Path, list[ScorecardRow], dict[str, Any]]:
    """Write both files. Gives back the folder, the rows and the shared facts."""
    rows, facts = build_scorecard(run_folders)
    folder = make_scorecard_folder(reports_dir)
    (folder / SCORECARD_HTML_FILE_NAME).write_text(
        build_scorecard_html(rows, facts), encoding="utf-8"
    )
    (folder / SCORECARD_MARKDOWN_FILE_NAME).write_text(
        scorecard_markdown(rows, facts), encoding="utf-8"
    )
    return folder, rows, facts


__all__ = [
    "SCORECARD_HTML_FILE_NAME",
    "SCORECARD_MARKDOWN_FILE_NAME",
    "ReportFolderError",
    "UnfairComparisonError",
    "build_scorecard",
    "make_scorecard_folder",
    "save_scorecard",
    "scorecard_folder_name",
    "scorecard_table",
]
