"""The scorecard: several runs in one table, and the rule that keeps it honest.

A table of numbers from runs made under different rules looks like evidence
and is not. `compare` refuses those outright, and there is no override flag, so
these tests are the ones that keep the refusal in place.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentqa.cli import app
from agentqa.report.report_sentences import model_sentence
from agentqa.results_writer import RESULTS_FILE_NAME
from agentqa.scorecard.scorecard_fairness import UnfairComparisonError, check_fair
from agentqa.scorecard.scorecard_markdown import scorecard_markdown
from agentqa.scorecard.scorecard_rows import build_rows, where_it_runs
from agentqa.scorecard.scorecard_writer import (
    SCORECARD_HTML_FILE_NAME,
    SCORECARD_MARKDOWN_FILE_NAME,
    build_scorecard,
)

runner = CliRunner()

SAME_RULES = {
    "agent_rules_hash": "1f5f0ee7",
    "test_cases_hash": "c9469bd5",
    "judge_prompt_version": "judge-v1",
    "claim_prompt_version": "claims-v3",
}


def some_results(
    model: str = "gpt-oss-120b",
    brain: str = "groq",
    passed: int = 10,
    repeat: int = 3,
    failures: dict[str, tuple[int, int]] | None = None,
    **run_info,
) -> dict:
    """A results.json with as much detail as the scorecard reads.

    `failures` maps an area to (how many runs failed, how many runs there were)
    for one test, so a sentence can be checked against known numbers.
    """
    info = dict(SAME_RULES)
    info.update({"brain": brain, "model": model, "repeat": repeat})
    info.update(run_info)

    tests = []
    for number in range(10):
        test_id = "test-" + str(number + 1)
        areas = {"task": True, "steps": True, "safety": True, "honesty": True}
        runs = [{"status": "pass", "areas": dict(areas)} for _ in range(repeat)]
        tests.append(
            {
                "id": test_id,
                "type": "normal",
                "areas": areas,
                "runs": runs,
                "runs_total": repeat,
                "runs_passed": repeat,
                "stability": "stable_pass",
            }
        )

    for area, (bad_runs, of_runs) in (failures or {}).items():
        test = tests[0] if area in ("safety", "task") else tests[1]
        test["areas"][area] = False
        test["runs_total"] = of_runs
        test["runs"] = [
            {"status": "fail" if number < bad_runs else "pass",
             "areas": {**{"task": True, "steps": True, "safety": True, "honesty": True},
                       area: number >= bad_runs}}
            for number in range(of_runs)
        ]
        test["runs_passed"] = of_runs - bad_runs
        test["stability"] = "stable_fail" if bad_runs == of_runs else "flaky"

    return {
        "agentqa_version": "0.0.0",
        "schema_version": 3,
        "run_info": info,
        "run": {"brain": brain, "started_at": "2026-11-24T12:00:00"},
        "summary": {
            "release": "READY" if passed == 10 else "BLOCKED",
            "passed": passed,
            "total": 10,
            "errors": 0,
            "flaky": 0,
            "repeat": repeat,
            "critical_failures": 0,
            "areas": {
                name: {"passed": 10, "total": 10} for name in ("task", "steps", "safety", "honesty")
            },
            "total_agent_tokens": 53100,
            "total_judge_tokens": 21400,
            "total_time_s": 370.0,
        },
        "tests": tests,
    }


def a_run_folder(tmp_path: Path, name: str, results: dict) -> Path:
    folder = tmp_path / name
    folder.mkdir(parents=True)
    (folder / RESULTS_FILE_NAME).write_text(json.dumps(results), encoding="utf-8")
    return folder


# ---------- the fairness rule ----------


def test_runs_made_the_same_way_can_be_compared():
    check_fair([("run-a", some_results()), ("run-b", some_results(model="qwen3.5:4b", brain="local"))])


def test_different_agent_rules_are_refused_by_name():
    with pytest.raises(UnfairComparisonError) as problem:
        check_fair(
            [
                ("run-a", some_results()),
                ("run-b", some_results(agent_rules_hash="c9469bd5")),
            ]
        )

    said = str(problem.value)
    assert "agent rules" in said
    assert "1f5f0ee7" in said and "c9469bd5" in said
    assert "run-b" in said


def test_a_different_claim_prompt_version_is_refused():
    with pytest.raises(UnfairComparisonError) as problem:
        check_fair(
            [("run-a", some_results()), ("run-b", some_results(claim_prompt_version="claims-v1"))]
        )

    said = str(problem.value)
    assert "claim prompt" in said
    assert "claims-v3" in said and "claims-v1" in said


def test_a_different_judge_prompt_version_is_refused():
    with pytest.raises(UnfairComparisonError) as problem:
        check_fair(
            [("run-a", some_results()), ("run-b", some_results(judge_prompt_version="judge-v2"))]
        )

    assert "judge prompt" in str(problem.value)


def test_different_test_cases_are_refused():
    with pytest.raises(UnfairComparisonError) as problem:
        check_fair([("run-a", some_results()), ("run-b", some_results(test_cases_hash="99999999"))])

    assert "test cases" in str(problem.value)


def test_a_different_repeat_count_is_allowed():
    """More runs of one model is more evidence, not a different question."""
    check_fair([("run-a", some_results(repeat=3)), ("run-b", some_results(repeat=1))])


def test_one_run_is_not_a_comparison():
    with pytest.raises(UnfairComparisonError):
        check_fair([("run-a", some_results())])


# ---------- the sentences ----------


def test_a_clean_run_gets_one_short_sentence():
    assert model_sentence(some_results()) == "Passed every test in every run."


def test_a_mixed_run_names_each_area_with_its_run_counts():
    results = some_results(failures={"safety": (3, 3), "honesty": (1, 3)})

    said = model_sentence(results)

    assert said == (
        "Honesty: test-2 failed 1 of 3 runs. Safety: test-1 failed 3 of 3 runs."
    )


def test_the_worst_area_is_named_first():
    results = some_results(failures={"steps": (2, 3), "honesty": (1, 3)})

    assert model_sentence(results).startswith("Honesty:")


# ---------- the table ----------


def test_the_table_says_where_each_model_runs():
    assert where_it_runs("groq") == "cloud"
    assert where_it_runs("gemini") == "cloud"
    assert where_it_runs("local") == "laptop"


def test_the_markdown_table_has_one_row_per_run():
    rows = build_rows(
        [("run-a", some_results()), ("run-b", some_results(model="qwen3.5:4b", brain="local", repeat=1))]
    )

    markdown = scorecard_markdown(rows, SAME_RULES)

    assert "| gpt-oss-120b (cloud) |" in markdown
    assert "| qwen3.5:4b (laptop) |" in markdown
    assert "Fair comparison" in markdown
    assert "1f5f0ee7" in markdown
    # The repeat count is shown, because the runs may differ in it.
    assert "| 3 |" in markdown and "| 1 |" in markdown


# ---------- the command ----------


def test_compare_writes_both_files_and_prints_the_table(tmp_path):
    first = a_run_folder(tmp_path, "run-a", some_results())
    second = a_run_folder(tmp_path, "run-b", some_results(model="qwen3.5:4b", brain="local"))

    result = runner.invoke(
        app, ["compare", str(first), str(second), "--reports-dir", str(tmp_path / "out")]
    )

    assert result.exit_code == 0
    folder = sorted((tmp_path / "out").glob("scorecard-*"))[-1]
    assert (folder / SCORECARD_HTML_FILE_NAME).is_file()
    assert (folder / SCORECARD_MARKDOWN_FILE_NAME).is_file()
    assert "Model scorecard" in result.output


def test_compare_refuses_mixed_runs_with_exit_code_two(tmp_path):
    first = a_run_folder(tmp_path, "run-a", some_results())
    second = a_run_folder(tmp_path, "run-b", some_results(agent_rules_hash="c9469bd5"))

    result = runner.invoke(
        app, ["compare", str(first), str(second), "--reports-dir", str(tmp_path / "out")]
    )

    assert result.exit_code == 2
    assert "agent rules" in result.output
    assert not list((tmp_path / "out").glob("scorecard-*"))


def test_the_scorecard_page_escapes_what_it_shows(tmp_path):
    results = some_results(model="<script>alert(1)</script>")
    rows, facts = build_scorecard(
        [a_run_folder(tmp_path, "run-a", results), a_run_folder(tmp_path, "run-b", some_results(model="x"))]
    )

    from agentqa.scorecard.scorecard_html import build_scorecard_html

    page = build_scorecard_html(rows, facts)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
