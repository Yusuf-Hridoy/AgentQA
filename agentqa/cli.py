"""The agentqa command.

    agentqa run --brain fake
    agentqa run --brain fake --only printer-ticket
    agentqa run --brain fake --cases test_cases/

Exit codes: 0 when the release is READY, 1 when it is BLOCKED, 2 for a setup error.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.text import Text

from agentqa import __version__
from agentqa.brains.brain_factory import BRAIN_FAKE, UnknownBrainError, build_brain
from agentqa.brains.fake_brain import FakeBrainScriptError
from agentqa.project_paths import PROJECT_ROOT, TEST_CASES_DIR
from agentqa.release_decision import decide_release
from agentqa.results_writer import RESULTS_FILE_NAME, save_run
from agentqa.run_all_tests import TestOutcome, run_one_test
from agentqa.test_case_loader import TestCaseFileError, load_test_cases

SETUP_ERROR_EXIT_CODE = 2

app = typer.Typer(add_completion=False, help="AgentQA: a testing tool for AI agents.")
console = Console()


def _dot() -> str:
    """Use a dot when the terminal speaks UTF-8, else a plain dash."""
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower().replace("-", "")
    if encoding in ("utf8", "utf_8"):
        return "·"
    return "-"


def _short_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _print_one_result(outcome: TestOutcome) -> None:
    word = "PASS" if outcome.passed else "FAIL"
    colour = "green" if outcome.passed else "red"
    line = Text("  ")
    line.append(word, style="bold " + colour)
    line.append("  " + outcome.test_case.id.ljust(24))
    if not outcome.passed:
        extra = outcome.first_failure_message
        if outcome.is_critical_failure:
            extra = "SAFETY " + _dot() + " " + extra
        line.append(extra, style=colour)
    console.print(line)


@app.callback()
def main() -> None:
    """AgentQA checks that an AI agent follows the rules and does its job."""


@app.command()
def run(
    brain: str = typer.Option(BRAIN_FAKE, "--brain", help="Which brain to test. Phase 1 has: fake."),
    only: str = typer.Option(None, "--only", help="Run one test case by its id."),
    cases: Path = typer.Option(TEST_CASES_DIR, "--cases", help="Folder with the test case files."),
    reports_dir: Path = typer.Option(None, "--reports-dir", help="Where to save the run folder."),
) -> None:
    """Run the AgentQA test cases against an agent."""
    started_at = datetime.now()

    try:
        test_cases = load_test_cases(cases, only=only)
    except TestCaseFileError as problem:
        console.print(Text(str(problem), style="red"))
        raise typer.Exit(code=SETUP_ERROR_EXIT_CODE)

    # Check the brain before the first test, so a wrong name fails fast.
    try:
        for test_case in test_cases:
            build_brain(brain, test_case.id)
    except (UnknownBrainError, FakeBrainScriptError) as problem:
        console.print(Text(str(problem), style="red"))
        raise typer.Exit(code=SETUP_ERROR_EXIT_CODE)

    dot = _dot()
    console.print("")
    console.print(
        Text("AgentQA " + dot + " " + str(len(test_cases)) + " tests " + dot + " brain: " + brain, style="bold")
    )
    console.print("")

    outcomes = []
    for test_case in test_cases:
        outcome = run_one_test(test_case, brain_name=brain)
        outcomes.append(outcome)
        _print_one_result(outcome)

    decision = decide_release(outcomes)
    finished_at = datetime.now()

    run_folder = save_run(
        outcomes=outcomes,
        decision=decision,
        brain_name=brain,
        started_at=started_at,
        finished_at=finished_at,
        cases_folder=_short_path(Path(cases)),
        reports_dir=reports_dir,
    )

    areas = decision.areas
    console.print("")
    console.print(
        "Task: " + areas["task"].as_text()
        + "   Steps: " + areas["steps"].as_text()
        + "   Safety: " + areas["safety"].as_text()
        + "   Critical failures: " + str(decision.critical_failures)
    )
    result_style = "bold green" if decision.is_ready else "bold red"
    console.print(
        Text(
            "Result: " + str(decision.passed_tests) + " of " + str(decision.total_tests)
            + " passed " + dot + " Release: " + decision.status,
            style=result_style,
        )
    )
    console.print("Report: " + _short_path(run_folder / RESULTS_FILE_NAME))
    console.print("")

    raise typer.Exit(code=decision.exit_code())


@app.command()
def version() -> None:
    """Show the AgentQA version."""
    console.print("AgentQA " + __version__)


if __name__ == "__main__":
    app()
