"""Saves what a run produced.

reports/run-<date>_<time>/
    results.json              the whole run
    report.html               the same run as a page a person can read
    traces/<test id>.json     one trace per test, when each test ran once
    traces/<test id>/run-1.json   one trace per run, when a test ran more than once

results.json holds a run_info block, so two results can be compared with no
guessing: which model answered, which judge judged, and whether the agent rules
or the test cases changed between the runs.

No API key is ever written here. Only names, models and hashes.

The HTML report and the scorecard are both built from results.json, so keep it
tidy.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agentqa import __version__
from agentqa.project_paths import REPORTS_DIR
from agentqa.release_decision import ReleaseDecision
from agentqa.repeat_runs import trace_path_for

RESULTS_FILE_NAME = "results.json"
TRACES_FOLDER_NAME = "traces"
RESULTS_SCHEMA_VERSION = 3


def run_folder_name(when: datetime | None = None) -> str:
    moment = when or datetime.now()
    return "run-" + moment.strftime("%Y-%m-%d_%H-%M-%S")


def make_run_folder(reports_dir: str | Path | None = None, when: datetime | None = None) -> Path:
    base = Path(reports_dir) if reports_dir else REPORTS_DIR
    folder = base / run_folder_name(when)
    counter = 2
    while folder.exists():
        folder = base / (run_folder_name(when) + "-" + str(counter))
        counter += 1
    (folder / TRACES_FOLDER_NAME).mkdir(parents=True, exist_ok=True)
    return folder


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _runs_of(outcome: Any) -> list[Any]:
    """The runs behind one test. A single outcome is a list of one."""
    runs = getattr(outcome, "runs", None)
    return list(runs) if runs else [outcome]


def write_traces(run_folder: Path, outcomes: list[Any]) -> list[Path]:
    """One trace file per run, at the path results.json points to."""
    written = []
    traces_folder = run_folder / TRACES_FOLDER_NAME
    traces_folder.mkdir(parents=True, exist_ok=True)
    for outcome in outcomes:
        runs = _runs_of(outcome)
        for number, one in enumerate(runs, start=1):
            path = run_folder / trace_path_for(outcome.test_case.id, number, len(runs))
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(path, one.trace.to_dict())
            written.append(path)
    return written


def build_results(
    outcomes: list[Any],
    decision: ReleaseDecision,
    brain_name: str,
    started_at: datetime,
    finished_at: datetime,
    cases_folder: str | None = None,
    run_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "agentqa_version": __version__,
        "schema_version": RESULTS_SCHEMA_VERSION,
        "run_info": dict(run_info or {}),
        "run": {
            "brain": brain_name,
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_ms": round((finished_at - started_at).total_seconds() * 1000, 3),
            "test_count": len(outcomes),
            "cases_folder": cases_folder,
        },
        "summary": decision.to_dict(),
        "tests": [outcome.to_dict() for outcome in outcomes],
    }


def save_run(
    outcomes: list[Any],
    decision: ReleaseDecision,
    brain_name: str,
    started_at: datetime,
    finished_at: datetime,
    cases_folder: str | None = None,
    reports_dir: str | Path | None = None,
    run_info: dict[str, Any] | None = None,
) -> Path:
    """Write results.json and one trace per test. Gives back the run folder."""
    folder = make_run_folder(reports_dir)
    write_traces(folder, outcomes)
    results = build_results(
        outcomes,
        decision,
        brain_name,
        started_at,
        finished_at,
        cases_folder,
        run_info=run_info,
    )
    _write_json(folder / RESULTS_FILE_NAME, results)
    return folder
