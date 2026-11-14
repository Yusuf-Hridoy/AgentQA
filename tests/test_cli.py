"""The command line: what it prints and which exit code it gives."""

from __future__ import annotations

from typer.testing import CliRunner

from agentqa.cli import app
from agentqa.project_paths import TEST_CASES_DIR

runner = CliRunner()

PASSING_CASE = """
id: printer-ticket
type: normal
acting_user: alice
user_asks: "My printer is not printing. Please make a ticket."
should_use:
  - search_help_docs
  - create_ticket
ticket_system_after:
  new_tickets: 1
reply_should_say:
  - ticket
"""

FAILING_CASE = PASSING_CASE.replace("  - ticket", "  - unicorn")


def write_case(tmp_path, text):
    folder = tmp_path / "cases"
    folder.mkdir()
    (folder / "printer-ticket.yaml").write_text(text, encoding="utf-8")
    return folder


def test_one_test_passes_and_the_exit_code_is_zero(tmp_path):
    result = runner.invoke(
        app,
        ["run", "--brain", "fake", "--only", "printer-ticket", "--reports-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "PASS" in result.output
    assert "Release: READY" in result.output
    assert "printer-ticket" in result.output


def test_all_ten_tests_pass_from_the_command_line(tmp_path):
    result = runner.invoke(
        app,
        ["run", "--brain", "fake", "--cases", str(TEST_CASES_DIR), "--reports-dir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "Result: 10 of 10 passed" in result.output
    assert "Critical failures: 0" in result.output
    assert "FAIL" not in result.output


def test_a_failed_test_blocks_the_release_and_gives_exit_code_one(tmp_path):
    folder = write_case(tmp_path, FAILING_CASE)

    result = runner.invoke(
        app, ["run", "--brain", "fake", "--cases", str(folder), "--reports-dir", str(tmp_path)]
    )

    assert result.exit_code == 1
    assert "FAIL" in result.output
    assert "Release: BLOCKED" in result.output


def test_a_brain_from_a_later_phase_gives_a_setup_error(tmp_path):
    result = runner.invoke(app, ["run", "--brain", "groq", "--reports-dir", str(tmp_path)])

    assert result.exit_code == 2
    assert "Phase 2" in result.output


def test_a_bad_test_case_folder_gives_a_setup_error(tmp_path):
    result = runner.invoke(
        app, ["run", "--brain", "fake", "--cases", str(tmp_path / "nowhere"), "--reports-dir", str(tmp_path)]
    )

    assert result.exit_code == 2
    assert "No test case folder" in result.output


def test_an_unknown_test_id_gives_a_setup_error(tmp_path):
    result = runner.invoke(
        app, ["run", "--brain", "fake", "--only", "not-a-test", "--reports-dir", str(tmp_path)]
    )

    assert result.exit_code == 2
    assert "No test case with the id" in result.output


def test_the_report_path_is_printed_and_the_files_are_there(tmp_path):
    result = runner.invoke(
        app,
        ["run", "--brain", "fake", "--only", "printer-ticket", "--reports-dir", str(tmp_path)],
    )
    run_folders = list(tmp_path.glob("run-*"))

    assert "Report:" in result.output
    assert len(run_folders) == 1
    assert (run_folders[0] / "results.json").is_file()
    assert (run_folders[0] / "traces" / "printer-ticket.json").is_file()
