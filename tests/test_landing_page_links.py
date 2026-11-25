"""Every link on the landing page has to work once the site is published.

GitHub Pages serves the site from docs/, so docs/index.html is the site root.
A link to ../anything leaves the site and is a 404 for every visitor. The page
once linked ../FINDINGS.md, which was exactly that.
"""

from __future__ import annotations

import re

from agentqa.publish.site_index import FINDINGS_URL, REPO_URL, write_index

HREF = re.compile(r'href="([^"]*)"')


def links_on(page: str) -> list[str]:
    return HREF.findall(page)


def page_with_everything(tmp_path):
    """A docs folder with a report, a scorecard and a saved trace in it."""
    docs = tmp_path / "docs"
    (docs / "reports" / "groq-repeat-3").mkdir(parents=True)
    (docs / "reports" / "groq-repeat-3" / "report.html").write_text("<html></html>", encoding="utf-8")
    (docs / "scorecard").mkdir(parents=True)
    (docs / "scorecard" / "scorecard.html").write_text("<html></html>", encoding="utf-8")
    (docs / "evidence").mkdir(parents=True)
    (docs / "evidence" / "hijacked-run.json").write_text("{}", encoding="utf-8")
    return write_index(docs).read_text(encoding="utf-8")


def test_no_link_on_the_landing_page_climbs_out_of_the_site(tmp_path):
    page = page_with_everything(tmp_path)
    climbing = [one for one in links_on(page) if one.startswith("../")]
    assert climbing == [], climbing


def test_the_findings_link_points_at_the_repository(tmp_path):
    page = page_with_everything(tmp_path)
    assert FINDINGS_URL in links_on(page)
    assert FINDINGS_URL.startswith(REPO_URL + "/blob/main/")
    assert FINDINGS_URL.endswith("/FINDINGS.md")


def test_every_other_link_is_either_inside_the_site_or_a_full_address(tmp_path):
    page = page_with_everything(tmp_path)
    for one in links_on(page):
        if one.startswith("http://") or one.startswith("https://"):
            continue
        assert not one.startswith("/"), one
        assert not one.startswith("../"), one


def test_the_published_landing_page_in_docs_has_no_climbing_links():
    """The page in the repository, not a page a test just built."""
    from agentqa.project_paths import DOCS_DIR

    index = DOCS_DIR / "index.html"
    if not index.is_file():
        return
    climbing = [one for one in links_on(index.read_text(encoding="utf-8")) if one.startswith("../")]
    assert climbing == [], climbing
