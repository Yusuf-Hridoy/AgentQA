"""The scorecard as Markdown, ready to paste into the README.

The same table and the same sentences as the HTML page, so a reader who only
sees the README is not reading a different story.
"""

from __future__ import annotations

from typing import Any

from agentqa.scorecard.scorecard_rows import ScorecardRow

HEADINGS = (
    "Model",
    "Agent",
    "Passed every run",
    "Task",
    "Steps",
    "Safety",
    "Honesty",
    "Guard",
    "Flaky",
    "Runs each",
    "Time",
    "Tokens",
)


def row_cells(row: ScorecardRow) -> list[str]:
    return [
        row.model_and_where,
        row.agent,
        row.passed_text,
        row.task,
        row.steps,
        row.safety,
        row.honesty,
        row.guard,
        str(row.flaky),
        str(row.repeat),
        str(round(row.time_s)) + " s",
        format(row.tokens, ","),
    ]


def scorecard_table(rows: list[ScorecardRow]) -> str:
    lines = [
        "| " + " | ".join(HEADINGS) + " |",
        "|" + "|".join([" --- "] * len(HEADINGS)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row_cells(row)) + " |")
    return "\n".join(lines)


def fairness_line(facts: dict[str, Any]) -> str:
    return (
        "Fair comparison: the same agent rules (`"
        + str(facts.get("agent_rules_hash") or "none recorded")
        + "`) and the same test cases (`"
        + str(facts.get("test_cases_hash") or "none recorded")
        + "`) in every run."
    )


def scorecard_markdown(rows: list[ScorecardRow], facts: dict[str, Any]) -> str:
    """The whole scorecard as Markdown."""
    said = "\n".join(
        "- **" + row.model_and_where + "**, agent " + row.agent + ". " + row.sentence
        for row in rows
    )
    return (
        "## Model scorecard\n\n"
        + fairness_line(facts) + "\n\n"
        + scorecard_table(rows) + "\n\n"
        + said + "\n"
    )
