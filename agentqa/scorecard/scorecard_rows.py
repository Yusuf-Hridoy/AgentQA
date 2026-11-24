"""One row per run: what the scorecard table and the Markdown both read from.

Nothing here decides anything. It reads results.json and puts the numbers in
one shape, so the HTML and the Markdown cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentqa.checks.check_result import AREA_HONESTY, AREA_SAFETY, AREA_STEPS, AREA_TASK
from agentqa.report.report_sentences import model_sentence

# Where a model answered from. A laptop and a data centre are not the same
# thing to run, and the table should say which it was.
CLOUD_BRAINS = ("groq", "gemini")
LAPTOP_BRAINS = ("local",)


@dataclass(frozen=True)
class ScorecardRow:
    """One run, ready to print."""

    folder: str
    model: str
    where: str
    passed: int
    total: int
    task: str
    steps: str
    safety: str
    honesty: str
    flaky: int
    repeat: int
    time_s: float
    tokens: int
    sentence: str
    release: str

    @property
    def model_and_where(self) -> str:
        return self.model + " (" + self.where + ")" if self.where else self.model

    @property
    def passed_text(self) -> str:
        return str(self.passed) + "/" + str(self.total)


def where_it_runs(brain: str | None) -> str:
    name = str(brain or "").lower()
    if name in CLOUD_BRAINS:
        return "cloud"
    if name in LAPTOP_BRAINS:
        return "laptop"
    if name == "fake":
        return "no model"
    return ""


def _area_text(summary: dict[str, Any], area: str) -> str:
    count = (summary.get("areas") or {}).get(area) or {}
    return str(count.get("passed", 0)) + "/" + str(count.get("total", 0))


def build_row(folder_name: str, results: dict[str, Any]) -> ScorecardRow:
    info = results.get("run_info") or {}
    summary = results.get("summary") or {}
    return ScorecardRow(
        folder=folder_name,
        model=str(info.get("model") or info.get("brain") or "unknown"),
        where=where_it_runs(info.get("brain")),
        passed=int(summary.get("passed") or 0),
        total=int(summary.get("total") or 0),
        task=_area_text(summary, AREA_TASK),
        steps=_area_text(summary, AREA_STEPS),
        safety=_area_text(summary, AREA_SAFETY),
        honesty=_area_text(summary, AREA_HONESTY),
        flaky=int(summary.get("flaky") or 0),
        repeat=int(summary.get("repeat") or info.get("repeat") or 1),
        time_s=round(float(summary.get("total_time_s") or 0), 1),
        tokens=int(summary.get("total_agent_tokens") or 0)
        + int(summary.get("total_judge_tokens") or 0),
        sentence=model_sentence(results),
        release=str(summary.get("release") or ""),
    )


def build_rows(runs: list[tuple[str, dict[str, Any]]]) -> list[ScorecardRow]:
    return [build_row(name, results) for name, results in runs]


def folder_label(folder: str | Path) -> str:
    """The run folder's own name, which is what a person typed."""
    return Path(folder).name or str(folder)
