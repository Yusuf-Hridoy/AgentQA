"""The scorecard as one self-contained page.

Same rules as the report: inline CSS, no network, escape everything, readable
on a phone and in dark mode.
"""

from __future__ import annotations

from typing import Any

from agentqa.report.report_html import safe
from agentqa.report.report_styles import REPORT_CSS
from agentqa.scorecard.scorecard_markdown import HEADINGS, fairness_line, row_cells
from agentqa.scorecard.scorecard_rows import ScorecardRow

PAGE_TITLE = "AgentQA model scorecard"

BANNER_CSS = """
.fair {
  margin-top: 20px;
  background: var(--pass-soft);
  border: 1px solid var(--pass);
  color: var(--ink);
  border-radius: 12px;
  padding: 12px 16px;
  font-size: 0.95rem;
}
.said { margin-top: 24px; }
.said div { margin-bottom: 10px; }
.said strong { font-family: var(--heading); }
"""


def _table(rows: list[ScorecardRow]) -> str:
    head = "".join("<th>" + safe(one) + "</th>" for one in HEADINGS)
    body = []
    for row in rows:
        cells = "".join("<td>" + safe(one) + "</td>" for one in row_cells(row))
        body.append("          <tr>" + cells + "</tr>")
    return (
        "    <div class=\"table-scroll\">\n"
        "      <table>\n"
        "        <thead>\n          <tr>" + head + "</tr>\n        </thead>\n"
        "        <tbody>\n" + "\n".join(body) + "\n        </tbody>\n"
        "      </table>\n"
        "    </div>\n"
    )


def build_scorecard_html(rows: list[ScorecardRow], facts: dict[str, Any]) -> str:
    said = "\n".join(
        "      <div><strong>" + safe(row.model_and_where) + "</strong> "
        + safe(row.sentence) + "</div>"
        for row in rows
    )
    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>" + safe(PAGE_TITLE) + "</title>\n"
        "<style>" + REPORT_CSS + BANNER_CSS + "</style>\n"
        "</head>\n"
        "<body>\n"
        "  <div class=\"wrap\">\n"
        "    <div class=\"masthead\">\n"
        "      <h1>AgentQA</h1>\n"
        "      <span class=\"what\">Model scorecard</span>\n"
        "    </div>\n"
        "    <div class=\"fair\">" + safe(fairness_line(facts)) + "</div>\n"
        "    <h2>Every run, side by side</h2>\n"
        + _table(rows)
        + "    <h2>What each model did</h2>\n"
        "    <div class=\"said\">\n" + said + "\n    </div>\n"
        "    <footer>Made by AgentQA</footer>\n"
        "  </div>\n"
        "</body>\n"
        "</html>\n"
    )
