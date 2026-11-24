"""Reads a run folder and writes report.html into it.

Every run writes its own report. `agentqa report <run folder>` builds one again
from a folder that is already saved, which is how an older run gets a page
without being run a second time.

The traces live in files next to results.json, so they are loaded and attached
to their run before the page is built. A missing trace file is not an error: a
report of what we have beats no report at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentqa.report.report_html import build_report_html
from agentqa.report.report_sentences import runs_of, tests_of
from agentqa.results_writer import RESULTS_FILE_NAME

REPORT_FILE_NAME = "report.html"


class ReportFolderError(Exception):
    """The folder is not a run folder, or its results.json cannot be read."""


def load_results(run_folder: str | Path) -> dict[str, Any]:
    """The results.json of one run folder, with a clear message when it is not there."""
    folder = Path(run_folder)
    if not folder.is_dir():
        raise ReportFolderError("No run folder at " + str(folder))

    results_file = folder / RESULTS_FILE_NAME
    if not results_file.is_file():
        raise ReportFolderError(
            "No " + RESULTS_FILE_NAME + " in " + str(folder)
            + ". A run folder is one that agentqa run made."
        )

    try:
        loaded = json.loads(results_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as problem:
        raise ReportFolderError("Could not read " + str(results_file) + ": " + str(problem)) from problem

    if not isinstance(loaded, dict):
        raise ReportFolderError("Bad " + str(results_file) + ": the file must hold an object.")
    return loaded


def attach_traces(results: dict[str, Any], run_folder: str | Path) -> dict[str, Any]:
    """Put each run's trace into the run, so the page can show the steps."""
    folder = Path(run_folder)
    for test in tests_of(results):
        for run in runs_of(test):
            if run.get("trace"):
                continue
            name = run.get("trace_file")
            if not name:
                continue
            path = folder / str(name)
            if not path.is_file():
                continue
            try:
                run["trace"] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    return results


def build_report_for_folder(run_folder: str | Path) -> str:
    """The page for one saved run folder, as a string."""
    folder = Path(run_folder)
    results = attach_traces(load_results(folder), folder)
    return build_report_html(results)


def write_report(run_folder: str | Path) -> Path:
    """Write report.html into the run folder. Gives back its path."""
    folder = Path(run_folder)
    page = build_report_for_folder(folder)
    path = folder / REPORT_FILE_NAME
    path.write_text(page, encoding="utf-8")
    return path
