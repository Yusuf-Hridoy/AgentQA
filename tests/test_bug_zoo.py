"""The bug zoo: eight agents broken on purpose, each caught by the right check.

If one of these walks through, AgentQA has a blind spot, and these tests fail
instead of a run looking green.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from agentqa.cli import app
from agentqa.project_paths import BUG_ZOO_DIR
from agentqa.zoo.zoo_loader import ZooFileError, load_broken_agents
from agentqa.zoo.zoo_results_writer import ZOO_RESULTS_FILE_NAME, save_zoo_run
from agentqa.zoo.zoo_runner import (
    CAUGHT,
    CAUGHT_BY_THE_WRONG_CHECK,
    MISSED,
    BrokenAgent,
    run_broken_agent,
    run_zoo,
    zoo_exit_code,
)

runner = CliRunner()

EXPECTED_AGENTS = {
    "chatty-notifier",
    "double-ticket",
    "endless-retry",
    "fake-success",
    "guesser",
    "half-truth",
    "hijacked",
    "snoop",
}


def test_every_broken_agent_is_loaded():
    agents = load_broken_agents()

    assert {agent.name for agent in agents} == EXPECTED_AGENTS
    for agent in agents:
        assert agent.bug and agent.based_on
        assert agent.script_file.is_file()


@pytest.mark.parametrize("agent", load_broken_agents(), ids=lambda one: one.name)
def test_every_broken_agent_is_caught_by_the_check_that_should_catch_it(agent):
    result = run_broken_agent(agent)

    assert result.result == CAUGHT, (
        agent.name + " was not caught by " + agent.expected_check
        + ". Failed checks: " + ", ".join(result.failed_checks)
    )
    assert agent.expected_check in result.failed_checks
    assert result.catching_message


def test_the_whole_zoo_is_caught():
    results = run_zoo()

    assert len(results) == len(EXPECTED_AGENTS)
    assert all(one.is_caught for one in results)
    assert zoo_exit_code(results) == 0


def test_the_two_honesty_agents_are_caught_by_claim_vs_reality():
    results = {one.agent.name: one for one in run_zoo()}

    assert results["fake-success"].catching_message == "Says it did create_ticket, but it never did."
    assert results["half-truth"].catching_message == "Says it did assign_ticket, but it never did."


def test_one_agent_can_be_run_alone():
    results = run_zoo(only="snoop")

    assert [one.agent.name for one in results] == ["snoop"]


def test_an_unknown_agent_name_is_an_error():
    with pytest.raises(ZooFileError):
        run_zoo(only="no-such-agent")


def test_a_wrong_expected_check_is_caught_by_the_wrong_check():
    """The bug is found, but not by the check that was meant to find it."""
    real = load_broken_agents(only="fake-success")[0]
    wrong = BrokenAgent(
        name=real.name,
        bug=real.bug,
        based_on=real.based_on,
        test_case=real.test_case,
        expected_check="reply_leaks_nothing",
        script_file=real.script_file,
        folder=real.folder,
    )

    result = run_broken_agent(wrong)

    assert result.result == CAUGHT_BY_THE_WRONG_CHECK
    assert zoo_exit_code([result]) == 1
    assert "claim_vs_reality" in result.failed_checks


def test_a_harmless_script_is_missed(tmp_path):
    """An agent that does the job right fails nothing, so the zoo says MISSED."""
    folder = tmp_path / "well-behaved"
    folder.mkdir()
    (folder / "about.yaml").write_text(
        "name: well-behaved\n"
        "bug: Nothing is broken here, so nothing can be caught.\n"
        "based_on: A test of the zoo itself.\n"
        "test_case: printer-ticket\n"
        "expected_check: claim_vs_reality\n",
        encoding="utf-8",
    )
    (folder / "script.yaml").write_text(
        (BUG_ZOO_DIR.parent / "fake_brain_scripts" / "printer-ticket.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    results = run_zoo(zoo_folder=tmp_path)

    assert [one.result for one in results] == [MISSED]
    assert zoo_exit_code(results) == 1


def test_a_zoo_folder_with_a_missing_file_is_an_error(tmp_path):
    (tmp_path / "broken").mkdir()

    with pytest.raises(ZooFileError):
        run_zoo(zoo_folder=tmp_path)


def test_the_zoo_results_are_saved(tmp_path):
    from datetime import datetime

    results = run_zoo(only="snoop")
    now = datetime.now()

    folder = save_zoo_run(results, now, now, reports_dir=tmp_path)

    import json

    saved = json.loads((folder / ZOO_RESULTS_FILE_NAME).read_text(encoding="utf-8"))
    assert saved["summary"] == {"caught": 1, "total": 1, "all_caught": True, "not_caught": []}
    assert saved["agents"][0]["name"] == "snoop"
    assert saved["agents"][0]["expected_check"] == "reply_leaks_nothing"
    assert saved["agents"][0]["result"] == CAUGHT


# ---------- the command ----------


def test_the_zoo_command_catches_everything_and_exits_zero(tmp_path):
    result = runner.invoke(app, ["zoo", "--reports-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "Caught 8 of 8" in result.output
    assert "no blind spots" in result.output
    assert "CAUGHT" in result.output


def test_the_result_word_and_the_agent_name_are_not_stuck_together(tmp_path):
    """The widest result word must still leave a gap before the agent name."""
    result = runner.invoke(app, ["zoo", "--reports-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "CAUGHTfake-success" not in result.output
    assert "CAUGHT  fake-success" in result.output


def test_the_zoo_command_can_run_one_agent(tmp_path):
    result = runner.invoke(app, ["zoo", "--only", "half-truth", "--reports-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "half-truth" in result.output
    assert "1 broken agent " in result.output


def test_the_zoo_command_exits_one_when_a_bug_walks_through(tmp_path):
    folder = tmp_path / "zoo" / "well-behaved"
    folder.mkdir(parents=True)
    (folder / "about.yaml").write_text(
        "name: well-behaved\n"
        "bug: Nothing is broken here.\n"
        "based_on: A test of the zoo itself.\n"
        "test_case: printer-ticket\n"
        "expected_check: claim_vs_reality\n",
        encoding="utf-8",
    )
    (folder / "script.yaml").write_text(
        (BUG_ZOO_DIR.parent / "fake_brain_scripts" / "printer-ticket.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["zoo", "--zoo-dir", str(tmp_path / "zoo"), "--reports-dir", str(tmp_path / "reports")],
    )

    assert result.exit_code == 1
    assert "MISSED" in result.output
    assert "blind spots: well-behaved" in result.output


def test_an_unknown_agent_name_gives_a_setup_error(tmp_path):
    result = runner.invoke(app, ["zoo", "--only", "no-such-agent", "--reports-dir", str(tmp_path)])

    assert result.exit_code == 2
    assert "no-such-agent" in result.output
