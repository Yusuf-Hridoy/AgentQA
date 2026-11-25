"""The agent and the guard, from results.json through to the page and the table.

A number that never leaves the terminal is not evidence. What the guard
stopped, and what it wrongly blocked, has to survive into results.json, into
report.html and into the scorecard, and it has to be escaped on the way.
"""

from __future__ import annotations

import importlib.util
import json

import pytest
from typer.testing import CliRunner

from agentqa.agents.agent_interface import AGENT_LANGGRAPH_GUARD, AGENT_LANGGRAPH_REACT
from agentqa.cli import SETUP_ERROR_EXIT_CODE, app
from agentqa.report.report_html import (
    GUARD_STOPPED_LABEL,
    GUARD_WRONGLY_BLOCKED_LABEL,
    build_report_html,
)
from agentqa.report.report_sentences import guard_note, guard_sentence
from agentqa.results_writer import RESULTS_FILE_NAME, RESULTS_SCHEMA_VERSION
from agentqa.run_info import build_run_info
from agentqa.scorecard.scorecard_fairness import UnfairComparisonError, check_fair
from agentqa.scorecard.scorecard_markdown import HEADINGS, row_cells, scorecard_markdown
from agentqa.scorecard.scorecard_rows import NO_GUARD, build_rows, guard_text
from tests.test_scorecard import a_run_folder, some_results

runner = CliRunner()

# Most of this file reads saved results, so it works with or without the
# extra. The three tests that really run a LangGraph agent need it installed.
HAS_LANGGRAPH = importlib.util.find_spec("langgraph") is not None
needs_langgraph = pytest.mark.skipif(not HAS_LANGGRAPH, reason="the LangGraph extra is not installed")


def guarded_results(stopped=None, wrongly_blocked=None, **run_info):
    """A results.json from a guard run, with the guard numbers filled in."""
    info = {
        "agent": AGENT_LANGGRAPH_GUARD,
        "agent_framework": "langgraph 1.2.20",
        "guard_prompt_version": "guard-v1",
    }
    info.update(run_info)
    results = some_results(**info)
    results["schema_version"] = RESULTS_SCHEMA_VERSION
    results["summary"]["guard_stopped"] = list(stopped or [])
    results["summary"]["guard_wrongly_blocked"] = list(wrongly_blocked or [])
    results["summary"]["total_guard_tokens"] = 4210
    return results


SIX_STOPPED = [
    {"tool": "send_notification", "reason": "The user did not ask for this.", "test_id": "t" + str(n)}
    for n in range(6)
]
ONE_WRONG = [{"tool": "create_ticket", "reason": "Not asked for.", "test_id": "t9"}]


# ---------- results.json ----------


def test_the_results_schema_is_version_four():
    assert RESULTS_SCHEMA_VERSION == 4


def test_run_info_records_the_agent_and_what_built_it():
    info = build_run_info(
        brain_name="fake",
        provider=None,
        model="fake",
        judge_mode="words",
        judge_model=None,
        judge_prompt_version=None,
        agent=AGENT_LANGGRAPH_REACT,
        agent_framework="langgraph 1.2.20",
    )
    assert info["agent"] == AGENT_LANGGRAPH_REACT
    assert info["agent_framework"] == "langgraph 1.2.20"
    assert info["guard_prompt_version"] is None


def test_a_builtin_run_says_it_used_no_framework():
    info = build_run_info(
        brain_name="fake",
        provider=None,
        model="fake",
        judge_mode="words",
        judge_model=None,
        judge_prompt_version=None,
    )
    assert info["agent"] == "builtin"
    assert info["agent_framework"] == "none"


@needs_langgraph
def test_a_real_guard_run_writes_the_guard_fields(tmp_path):
    result = runner.invoke(
        app,
        ["run", "--agent", AGENT_LANGGRAPH_GUARD, "--brain", "fake", "--reports-dir", str(tmp_path)],
    )
    assert result.exit_code == 0

    folder = sorted(tmp_path.glob("run-*"))[-1]
    saved = json.loads((folder / RESULTS_FILE_NAME).read_text(encoding="utf-8"))

    assert saved["schema_version"] == 4
    assert saved["run_info"]["agent"] == AGENT_LANGGRAPH_GUARD
    assert saved["run_info"]["agent_framework"].startswith("langgraph ")
    assert saved["run_info"]["guard_prompt_version"] == "guard-v1"
    assert saved["summary"]["guard_stopped"] == []
    assert saved["summary"]["guard_wrongly_blocked"] == []
    assert "total_guard_tokens" in saved["summary"]
    for test in saved["tests"]:
        assert "guard_tokens" in test
        assert test["guard_stopped"] == []
        assert test["guard_wrongly_blocked"] == []


@needs_langgraph
def test_the_terminal_says_what_the_guard_did(tmp_path):
    result = runner.invoke(
        app,
        ["run", "--agent", AGENT_LANGGRAPH_GUARD, "--brain", "fake", "--reports-dir", str(tmp_path)],
    )
    assert "agent: " + AGENT_LANGGRAPH_GUARD in result.output
    assert "Guard: stopped" in result.output
    assert "wrongly blocked" in result.output


@needs_langgraph
def test_an_agent_without_a_guard_prints_no_guard_line(tmp_path):
    result = runner.invoke(
        app,
        ["run", "--agent", AGENT_LANGGRAPH_REACT, "--brain", "fake", "--reports-dir", str(tmp_path)],
    )
    assert "agent: " + AGENT_LANGGRAPH_REACT in result.output
    assert "Guard:" not in result.output


# ---------- the sentences ----------


def test_the_guard_note_names_both_numbers():
    said = guard_note(guarded_results(SIX_STOPPED, ONE_WRONG))
    assert said == "Guard stopped 6 writes, wrongly blocked 1."


def test_a_run_without_a_guard_has_no_guard_note():
    assert guard_note(some_results()) == ""
    assert guard_sentence(some_results()) == ""


def test_the_guard_sentence_names_the_tools():
    said = guard_sentence(guarded_results(SIX_STOPPED, ONE_WRONG))
    assert "send_notification" in said
    assert "Wrongly blocked create_ticket." in said


def test_a_guard_that_stopped_nothing_still_says_so():
    said = guard_sentence(guarded_results([], []))
    assert "stopped nothing" in said
    assert "Wrongly blocked nothing." in said


# ---------- the report ----------


def test_the_report_header_has_an_agent_pill():
    page = build_report_html(guarded_results(SIX_STOPPED, ONE_WRONG))
    assert "langgraph-guard (langgraph 1.2.20)" in page


def test_the_safety_card_carries_the_guard_note():
    page = build_report_html(guarded_results(SIX_STOPPED, ONE_WRONG))
    assert "Guard stopped 6 writes, wrongly blocked 1." in page


def test_the_run_info_box_names_the_agent_and_the_guard_prompt():
    page = build_report_html(guarded_results())
    assert "<td>Agent</td>" in page
    assert "<td>langgraph-guard</td>" in page
    assert "<td>guard-v1</td>" in page


def a_results_with_one_guard_block(tool: str, wrongly: bool, reason: str):
    results = guarded_results([], [{"tool": tool, "reason": reason}] if wrongly else [])
    run = {
        "status": "fail",
        "areas": {"task": False, "steps": True, "safety": True, "honesty": True},
        "guard_wrongly_blocked": [{"tool": tool, "reason": reason}] if wrongly else [],
        "guard_stopped": [] if wrongly else [{"tool": tool, "reason": reason}],
        "trace": {
            "events": [
                {"step": 1, "kind": "guard_blocked", "tool": tool, "inputs": {"ticket_id": "TCK-1"}, "error": reason}
            ]
        },
        "checks": [],
    }
    results["tests"][0].update(
        {"runs": [run], "runs_total": 1, "runs_passed": 0, "stability": "stable_fail"}
    )
    return results


def test_a_guard_block_shows_in_the_steps_with_a_green_label():
    page = build_report_html(a_results_with_one_guard_block("send_notification", False, "Not asked for."))
    assert "guard_blocked" in page
    assert GUARD_STOPPED_LABEL in page
    assert "Not asked for." in page


def test_a_wrongly_blocked_action_gets_the_orange_label():
    page = build_report_html(a_results_with_one_guard_block("create_ticket", True, "Not asked for."))
    assert GUARD_WRONGLY_BLOCKED_LABEL in page


def test_a_guard_reason_from_a_model_is_still_escaped():
    """The reason comes back from a model, so it is never trusted as HTML."""
    nasty = "<script>alert('x')</script>"
    page = build_report_html(a_results_with_one_guard_block("create_ticket", True, nasty))
    assert nasty not in page
    assert "&lt;script&gt;" in page


# ---------- the scorecard ----------


def test_the_table_has_an_agent_column_and_a_guard_column():
    assert "Agent" in HEADINGS
    assert "Guard" in HEADINGS


def test_the_guard_cell_says_what_the_guard_did():
    assert guard_text(guarded_results(SIX_STOPPED, ONE_WRONG)) == "stopped 6 · wrong 1"


def test_a_run_with_no_guard_shows_a_dash():
    assert guard_text(some_results()) == NO_GUARD


def test_the_row_carries_the_agent_name():
    rows = build_rows([("run-a", guarded_results(SIX_STOPPED, ONE_WRONG))])
    assert rows[0].agent == AGENT_LANGGRAPH_GUARD
    cells = row_cells(rows[0])
    assert len(cells) == len(HEADINGS)
    assert AGENT_LANGGRAPH_GUARD in cells
    assert "stopped 6 · wrong 1" in cells


def test_the_markdown_row_says_what_the_guard_stopped():
    rows = build_rows(
        [
            ("run-a", some_results()),
            ("run-b", guarded_results(SIX_STOPPED, ONE_WRONG)),
        ]
    )
    said = scorecard_markdown(rows, {"agent_rules_hash": "1f5f0ee7", "test_cases_hash": "c9469bd5"})
    assert "| Agent |" in said
    assert "| Guard |" in said
    assert "agent builtin." in said
    assert "Guard: stopped send_notification" in said


# ---------- the fairness rule ----------


def test_two_agents_may_be_compared_because_the_agent_is_what_we_compare():
    check_fair(
        [
            ("run-builtin", some_results(agent="builtin")),
            ("run-react", some_results(agent=AGENT_LANGGRAPH_REACT)),
            ("run-guard", guarded_results()),
        ]
    )


def test_two_guard_runs_with_different_guard_prompts_are_refused():
    with pytest.raises(UnfairComparisonError) as problem:
        check_fair(
            [
                ("run-a", guarded_results()),
                ("run-b", guarded_results(guard_prompt_version="guard-v2")),
            ]
        )
    said = str(problem.value)
    assert "guard prompt" in said
    assert "guard-v1" in said and "guard-v2" in said
    assert "run-a" in said and "run-b" in said


def test_a_guard_run_next_to_a_run_with_no_guard_is_still_fair():
    """An agent with no guard has no guard prompt to differ about."""
    check_fair([("run-a", some_results()), ("run-b", guarded_results())])


def test_compare_stops_with_the_setup_code_on_mixed_guard_prompts(tmp_path):
    first = a_run_folder(tmp_path, "run-a", guarded_results())
    second = a_run_folder(tmp_path, "run-b", guarded_results(guard_prompt_version="guard-v2"))

    result = runner.invoke(app, ["compare", str(first), str(second), "--reports-dir", str(tmp_path)])

    assert result.exit_code == SETUP_ERROR_EXIT_CODE
    assert "guard prompt" in result.output
