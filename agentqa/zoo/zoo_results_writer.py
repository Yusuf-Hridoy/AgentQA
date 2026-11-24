"""Saves what a zoo run found.

reports/zoo-<date>_<time>/zoo_results.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agentqa import __version__
from agentqa.project_paths import REPORTS_DIR
from agentqa.zoo.zoo_runner import ZooResult, all_caught

ZOO_RESULTS_FILE_NAME = "zoo_results.json"


def zoo_folder_name(when: datetime | None = None) -> str:
    moment = when or datetime.now()
    return "zoo-" + moment.strftime("%Y-%m-%d_%H-%M-%S")


def make_zoo_folder(reports_dir: str | Path | None = None, when: datetime | None = None) -> Path:
    base = Path(reports_dir) if reports_dir else REPORTS_DIR
    folder = base / zoo_folder_name(when)
    counter = 2
    while folder.exists():
        folder = base / (zoo_folder_name(when) + "-" + str(counter))
        counter += 1
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def build_zoo_results(results: list[ZooResult], started_at: datetime, finished_at: datetime) -> dict[str, Any]:
    caught = [one for one in results if one.is_caught]
    return {
        "agentqa_version": __version__,
        "zoo": {
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "broken_agents": len(results),
        },
        "summary": {
            "caught": len(caught),
            "total": len(results),
            "all_caught": all_caught(results),
            "not_caught": [one.agent.name for one in results if not one.is_caught],
        },
        "agents": [one.to_dict() for one in results],
    }


def save_zoo_run(
    results: list[ZooResult],
    started_at: datetime,
    finished_at: datetime,
    reports_dir: str | Path | None = None,
) -> Path:
    """Write zoo_results.json. Gives back the zoo folder."""
    folder = make_zoo_folder(reports_dir)
    data = build_zoo_results(results, started_at, finished_at)
    (folder / ZOO_RESULTS_FILE_NAME).write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return folder
