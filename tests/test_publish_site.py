"""Publishing a run to docs/, and the key scan that happens first.

Publishing is the one step that cannot be taken back: a key pushed to a public
repository is burned whatever the next commit says. So the scan runs before
anything is copied, and a single hit stops the whole command.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agentqa.cli import app
from agentqa.publish.key_scanner import scan_folder, scan_text
from agentqa.publish.publish_run import KeyFoundError, publish_folder
from agentqa.publish.site_index import INDEX_FILE_NAME, NOJEKYLL_FILE_NAME, write_index
from agentqa.report.report_builder import REPORT_FILE_NAME
from agentqa.results_writer import RESULTS_FILE_NAME

runner = CliRunner()

FAKE_GROQ_KEY = "gsk_" + "a1b2c3d4" * 5
FAKE_GEMINI_KEY = "AIza" + "Sy0123456789abcdefghijklmnop"


def a_run_folder(tmp_path: Path, name: str = "run-2026-11-24_12-00-00") -> Path:
    folder = tmp_path / name
    (folder / "traces").mkdir(parents=True)
    (folder / RESULTS_FILE_NAME).write_text(
        json.dumps({"summary": {"release": "READY"}, "tests": []}), encoding="utf-8"
    )
    (folder / REPORT_FILE_NAME).write_text("<!doctype html><html></html>", encoding="utf-8")
    (folder / "traces" / "printer-ticket.json").write_text("{\"events\": []}", encoding="utf-8")
    return folder


# ---------- the scan ----------


def test_a_groq_key_is_spotted():
    assert scan_text("the key is " + FAKE_GROQ_KEY) == "a Groq key"


def test_a_gemini_key_is_spotted():
    assert scan_text("the key is " + FAKE_GEMINI_KEY) == "a Gemini key"


def test_ordinary_results_hold_no_key():
    assert scan_text('{"brain": "groq", "model": "gpt-oss-120b"}') is None


def test_a_key_set_in_the_environment_is_spotted_even_in_an_unknown_shape():
    found = scan_text("token=zzz-not-a-known-shape-zzz", [("SOME_API_KEY", "zzz-not-a-known-shape-zzz")])

    assert found == "the value of SOME_API_KEY"


def test_the_scan_never_gives_back_the_key(tmp_path):
    folder = a_run_folder(tmp_path)
    (folder / RESULTS_FILE_NAME).write_text(FAKE_GROQ_KEY, encoding="utf-8")

    found = scan_folder(folder)

    assert len(found) == 1
    assert FAKE_GROQ_KEY not in found[0].as_text()
    assert found[0].what == "a Groq key"


# ---------- publishing ----------


def test_a_run_is_copied_into_docs(tmp_path):
    source = a_run_folder(tmp_path)
    docs = tmp_path / "docs"

    target, copied, index = publish_folder(source, name="groq-repeat-3", docs_dir=docs)

    assert target == docs / "reports" / "groq-repeat-3"
    assert (target / REPORT_FILE_NAME).is_file()
    assert (target / RESULTS_FILE_NAME).is_file()
    assert (target / "traces" / "printer-ticket.json").is_file()
    assert len(copied) == 3
    assert index.is_file()


def test_a_folder_holding_a_key_is_refused_and_nothing_is_copied(tmp_path):
    source = a_run_folder(tmp_path)
    (source / "traces" / "printer-ticket.json").write_text(
        '{"note": "' + FAKE_GROQ_KEY + '"}', encoding="utf-8"
    )
    docs = tmp_path / "docs"

    try:
        publish_folder(source, name="leaky", docs_dir=docs)
        raise AssertionError("publish_folder should have refused")
    except KeyFoundError as problem:
        assert "Nothing was copied" in str(problem)
        assert FAKE_GROQ_KEY not in str(problem)

    assert not (docs / "reports").exists()


def test_the_publish_command_exits_two_on_a_key(tmp_path):
    source = a_run_folder(tmp_path)
    (source / RESULTS_FILE_NAME).write_text('{"k": "' + FAKE_GEMINI_KEY + '"}', encoding="utf-8")

    result = runner.invoke(
        app, ["publish", str(source), "--docs-dir", str(tmp_path / "docs")]
    )

    assert result.exit_code == 2
    assert "Nothing was copied" in result.output
    assert FAKE_GEMINI_KEY not in result.output


def test_a_scorecard_folder_goes_to_its_own_place(tmp_path):
    source = tmp_path / "scorecard-2026-11-24_12-00-00"
    source.mkdir()
    (source / "scorecard.html").write_text("<!doctype html><html></html>", encoding="utf-8")
    (source / "scorecard.md").write_text("# Model scorecard", encoding="utf-8")
    docs = tmp_path / "docs"

    target, _copied, _index = publish_folder(source, docs_dir=docs)

    assert target == docs / "scorecard"
    assert (target / "scorecard.html").is_file()


def test_publishing_the_same_name_twice_replaces_it(tmp_path):
    source = a_run_folder(tmp_path)
    docs = tmp_path / "docs"
    publish_folder(source, name="same", docs_dir=docs)
    (docs / "reports" / "same" / "left-over.json").write_text("{}", encoding="utf-8")

    publish_folder(source, name="same", docs_dir=docs)

    assert not (docs / "reports" / "same" / "left-over.json").exists()


# ---------- the landing page ----------


def test_the_index_links_what_was_published(tmp_path):
    source = a_run_folder(tmp_path)
    docs = tmp_path / "docs"
    publish_folder(source, name="groq-repeat-3", docs_dir=docs)

    page = (docs / INDEX_FILE_NAME).read_text(encoding="utf-8")

    assert "reports/groq-repeat-3/report.html" in page
    assert "github.com/Yusuf-Hridoy/AgentQA" in page
    assert "AgentQA tests an AI support agent" in page


def test_the_index_links_the_evidence_traces(tmp_path):
    docs = tmp_path / "docs"
    (docs / "evidence").mkdir(parents=True)
    (docs / "evidence" / "qwen-4b-obeys-hidden-instruction.json").write_text("{}", encoding="utf-8")

    page = write_index(docs).read_text(encoding="utf-8")

    assert "evidence/qwen-4b-obeys-hidden-instruction.json" in page
    assert "FINDINGS.md" in page


def test_the_index_says_so_when_nothing_is_published_yet(tmp_path):
    page = write_index(tmp_path / "docs").read_text(encoding="utf-8")

    assert "No report has been published yet" in page


def test_a_nojekyll_file_is_made(tmp_path):
    docs = tmp_path / "docs"
    write_index(docs)

    assert (docs / NOJEKYLL_FILE_NAME).is_file()


def test_the_index_needs_no_network(tmp_path):
    docs = tmp_path / "docs"
    publish_folder(a_run_folder(tmp_path), name="one", docs_dir=docs)

    page = (docs / INDEX_FILE_NAME).read_text(encoding="utf-8")

    assert "<link" not in page
    assert "@import" not in page
    # The only link out is the repository itself.
    assert page.count("https://") == 1
