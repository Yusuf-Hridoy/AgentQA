"""The landing page at docs/index.html.

It is rebuilt from what is actually in docs/, so a link on the page is always a
file that exists. Same rules as the report: one file, no network, dark mode,
readable on a phone.
"""

from __future__ import annotations

from pathlib import Path

from agentqa.report.report_html import safe
from agentqa.report.report_styles import REPORT_CSS

INDEX_FILE_NAME = "index.html"
NOJEKYLL_FILE_NAME = ".nojekyll"
REPORTS_FOLDER_NAME = "reports"
SCORECARD_FOLDER_NAME = "scorecard"
EVIDENCE_FOLDER_NAME = "evidence"

REPO_URL = "https://github.com/Yusuf-Hridoy/AgentQA"

WHAT_IT_IS = (
    "AgentQA tests an AI support agent the way a tester would: it gives the agent "
    "a job to do, watches every tool it calls, and checks what really changed.",
    "Each report below is a real run. Open one to see what the agent did, what it "
    "said it did, and where the two did not match.",
)

INDEX_CSS = """
.links { list-style: none; padding: 0; margin: 12px 0 0; }
.links li {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 12px 16px;
  margin-bottom: 8px;
}
.links a { font-weight: 600; text-decoration: none; }
.links a:hover { text-decoration: underline; }
.links .where { display: block; color: var(--ink-soft); font-size: 0.85rem; margin-top: 2px; }
.lede { font-size: 1.05rem; color: var(--ink-soft); max-width: 60ch; }
"""


def _link(href: str, title: str, note: str = "") -> str:
    return (
        "      <li><a href=\"" + safe(href) + "\">" + safe(title) + "</a>"
        + ("<span class=\"where\">" + safe(note) + "</span>" if note else "")
        + "</li>"
    )


def published_reports(docs_dir: str | Path) -> list[str]:
    """The names of the published run folders, in name order."""
    folder = Path(docs_dir) / REPORTS_FOLDER_NAME
    if not folder.is_dir():
        return []
    return sorted(one.name for one in folder.iterdir() if one.is_dir())


def published_scorecards(docs_dir: str | Path) -> list[str]:
    folder = Path(docs_dir) / SCORECARD_FOLDER_NAME
    if not folder.is_dir():
        return []
    if (folder / "scorecard.html").is_file():
        return ["scorecard.html"]
    return sorted(
        one.name + "/scorecard.html"
        for one in folder.iterdir()
        if one.is_dir() and (one / "scorecard.html").is_file()
    )


def evidence_files(docs_dir: str | Path) -> list[str]:
    folder = Path(docs_dir) / EVIDENCE_FOLDER_NAME
    if not folder.is_dir():
        return []
    return sorted(one.name for one in folder.iterdir() if one.is_file())


def build_index_html(docs_dir: str | Path) -> str:
    """The landing page for whatever is in docs/ right now."""
    docs = Path(docs_dir)

    reports = [
        _link(REPORTS_FOLDER_NAME + "/" + name + "/report.html", name, "A full test run")
        for name in published_reports(docs)
    ]
    scorecards = [
        _link(SCORECARD_FOLDER_NAME + "/" + name, "Model scorecard", "Several runs side by side")
        for name in published_scorecards(docs)
    ]
    evidence = [
        _link(EVIDENCE_FOLDER_NAME + "/" + name, name, "A saved trace")
        for name in evidence_files(docs)
    ]

    sections = []
    if reports or scorecards:
        sections.append(
            "    <h2>Reports</h2>\n    <ul class=\"links\">\n"
            + "\n".join(scorecards + reports) + "\n    </ul>\n"
        )
    else:
        sections.append(
            "    <h2>Reports</h2>\n"
            "    <p class=\"lede\">No report has been published yet. "
            "Run <code>agentqa publish</code> to add one.</p>\n"
        )

    if evidence:
        sections.append(
            "    <h2>Evidence</h2>\n    <ul class=\"links\">\n"
            + _link("../FINDINGS.md", "FINDINGS.md", "What AgentQA found, with the traces")
            + "\n" + "\n".join(evidence) + "\n    </ul>\n"
        )

    lede = "\n".join("    <p class=\"lede\">" + safe(one) + "</p>" for one in WHAT_IT_IS)

    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>AgentQA</title>\n"
        "<style>" + REPORT_CSS + INDEX_CSS + "</style>\n"
        "</head>\n"
        "<body>\n"
        "  <div class=\"wrap\">\n"
        "    <div class=\"masthead\">\n"
        "      <h1>AgentQA</h1>\n"
        "      <span class=\"what\">Testing for AI agents</span>\n"
        "    </div>\n"
        + lede + "\n"
        + "".join(sections)
        + "    <h2>The code</h2>\n    <ul class=\"links\">\n"
        + _link(REPO_URL, "AgentQA on GitHub", "How it works, and how to run it yourself")
        + "\n    </ul>\n"
        "    <footer>Made by AgentQA</footer>\n"
        "  </div>\n"
        "</body>\n"
        "</html>\n"
    )


def write_index(docs_dir: str | Path) -> Path:
    """Write docs/index.html and make sure docs/.nojekyll is there."""
    docs = Path(docs_dir)
    docs.mkdir(parents=True, exist_ok=True)
    nojekyll = docs / NOJEKYLL_FILE_NAME
    if not nojekyll.exists():
        nojekyll.write_text("", encoding="utf-8")
    path = docs / INDEX_FILE_NAME
    path.write_text(build_index_html(docs), encoding="utf-8")
    return path
