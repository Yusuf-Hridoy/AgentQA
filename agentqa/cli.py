"""The agentqa command.

    agentqa run --brain fake
    agentqa run --brain local
    agentqa run --brain groq --only printer-ticket
    agentqa run --brain gemini --judge words
    agentqa run --brain groq --no-cache
    agentqa check-judge

Exit codes: 0 when the release is READY, 1 when it is BLOCKED, 2 for a setup
error. Any error blocks the release: a release cannot be ready if we do not
know what the agent did.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.text import Text

from agentqa import __version__
from agentqa.brains.brain_errors import BrainSetupError
from agentqa.brains.brain_factory import (
    BRAIN_FAKE,
    UnknownBrainError,
    brain_model_name,
    check_brain_ready,
    is_real_brain,
)
from agentqa.brains.brain_presets import short_model_name
from agentqa.brains.fake_brain import FakeBrainScriptError
from agentqa.judge.judge_calibration_loader import JudgeExampleFileError, load_judge_examples
from agentqa.judge.judge_modes import (
    JUDGE_AUTO,
    JUDGE_GEMINI,
    UnknownJudgeModeError,
    build_judge,
    settled_judge_mode,
)
from agentqa.judge.judge_prompt import JUDGE_PROMPT_VERSION
from agentqa.judge.reply_judge import JudgeError
from agentqa.project_paths import PROJECT_ROOT, TEST_CASES_DIR
from agentqa.release_decision import decide_release
from agentqa.results_writer import RESULTS_FILE_NAME, save_run
from agentqa.run_all_tests import STATUS_ERROR, STATUS_PASS, TestOutcome, run_one_test
from agentqa.run_info import build_run_info
from agentqa.test_case_loader import TestCaseFileError, load_test_cases

SETUP_ERROR_EXIT_CODE = 2

# Keys come from the environment or from a .env file. Never from the code.
load_dotenv()
load_dotenv(PROJECT_ROOT / ".env")

app = typer.Typer(add_completion=False, help="AgentQA: a testing tool for AI agents.")
console = Console()

STATUS_WORD = {STATUS_PASS: "PASS", STATUS_ERROR: "ERROR"}
STATUS_COLOUR = {STATUS_PASS: "green", STATUS_ERROR: "yellow"}


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


def _thousands(number: int) -> str:
    return format(int(number), ",")


def _print_one_result(outcome: TestOutcome) -> None:
    """The word is always printed, not only the colour."""
    word = STATUS_WORD.get(outcome.status, "FAIL")
    colour = STATUS_COLOUR.get(outcome.status, "red")
    line = Text("  ")
    line.append(word.ljust(6), style="bold " + colour)
    line.append(" " + outcome.test_case.id.ljust(24))
    if outcome.status != STATUS_PASS:
        extra = outcome.first_failure_message
        if outcome.is_critical_failure:
            extra = "SAFETY " + _dot() + " " + extra
        line.append(extra, style=colour)
    console.print(line)


def _header_brain(brain: str) -> str:
    if not is_real_brain(brain):
        return brain
    return short_model_name(brain_model_name(brain)) + " (" + brain + ")"


def _stop_with_setup_error(problem: Exception) -> None:
    console.print(Text(str(problem), style="red"))
    raise typer.Exit(code=SETUP_ERROR_EXIT_CODE)


@app.callback()
def main() -> None:
    """AgentQA checks that an AI agent follows the rules and does its job."""


@app.command()
def run(
    brain: str = typer.Option(BRAIN_FAKE, "--brain", help="Which brain to test: fake, local, groq or gemini."),
    judge: str = typer.Option(
        JUDGE_AUTO,
        "--judge",
        help="How to check the reply: auto, gemini or words. auto uses words for the fake brain.",
    ),
    only: str = typer.Option(None, "--only", help="Run one test case by its id."),
    cases: Path = typer.Option(TEST_CASES_DIR, "--cases", help="Folder with the test case files."),
    reports_dir: Path = typer.Option(None, "--reports-dir", help="Where to save the run folder."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Do not reuse saved judge answers."),
) -> None:
    """Run the AgentQA test cases against an agent."""
    started_at = datetime.now()

    try:
        test_cases = load_test_cases(cases, only=only)
    except TestCaseFileError as problem:
        _stop_with_setup_error(problem)

    # Everything that can fail before the first test, fails before the first test.
    try:
        judge_mode = settled_judge_mode(judge, brain)
        check_brain_ready(brain, [case.id for case in test_cases])
        reply_judge = build_judge(judge_mode, use_cache=not no_cache)
    except (
        UnknownBrainError,
        UnknownJudgeModeError,
        BrainSetupError,
        FakeBrainScriptError,
    ) as problem:
        _stop_with_setup_error(problem)

    judge_model = reply_judge.model if reply_judge is not None else None
    dot = _dot()

    console.print("")
    console.print(
        Text(
            "AgentQA " + dot + " " + str(len(test_cases)) + " tests "
            + dot + " brain: " + _header_brain(brain)
            + " " + dot + " judge: " + (judge_model or "words"),
            style="bold",
        )
    )
    console.print("")

    outcomes = []
    for test_case in test_cases:
        outcome = run_one_test(test_case, brain_name=brain, judge=reply_judge)
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
        run_info=build_run_info(
            brain_name=brain,
            provider=brain if is_real_brain(brain) else None,
            model=brain_model_name(brain),
            judge_mode=judge_mode,
            judge_model=judge_model,
            judge_prompt_version=JUDGE_PROMPT_VERSION if reply_judge is not None else None,
            cases_folder=cases,
        ),
    )

    areas = decision.areas
    console.print("")
    console.print(
        "Task: " + areas["task"].as_text()
        + "   Steps: " + areas["steps"].as_text()
        + "   Safety: " + areas["safety"].as_text()
        + "   Critical failures: " + str(decision.critical_failures)
        + "   Errors: " + str(decision.errors)
    )
    console.print(
        "Agent tokens: " + _thousands(decision.total_agent_tokens)
        + " " + dot + " Judge tokens: " + _thousands(decision.total_judge_tokens)
        + " " + dot + " Time: " + str(round(decision.total_time_s)) + " s"
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


@app.command("check-judge")
def check_judge(
    examples: Path = typer.Option(None, "--examples", help="The judge example file to use."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Do not reuse saved judge answers."),
) -> None:
    """Measure the judge against hand-labelled examples."""
    try:
        labelled = load_judge_examples(examples)
        reply_judge = build_judge(JUDGE_GEMINI, use_cache=not no_cache)
    except (JudgeExampleFileError, BrainSetupError) as problem:
        _stop_with_setup_error(problem)

    dot = _dot()
    console.print("")
    console.print(
        Text(
            "AgentQA check-judge " + dot + " " + str(len(labelled)) + " examples "
            + dot + " judge: " + reply_judge.model + " " + dot + " " + JUDGE_PROMPT_VERSION,
            style="bold",
        )
    )
    console.print("")

    agreed = 0
    disagreements = []
    for example in labelled:
        try:
            verdict = reply_judge.judge(example.user_asks, example.reply, example.expectation)
        except JudgeError as problem:
            console.print(Text("  ERROR  " + example.id.ljust(32) + " " + str(problem), style="yellow"))
            disagreements.append((example, "no answer", str(problem)))
            continue

        said = "pass" if verdict.passed else "fail"
        if said == example.label:
            agreed += 1
            console.print(Text("  AGREE  ", style="bold green") + Text(example.id.ljust(32) + " " + said))
        else:
            disagreements.append((example, said, verdict.reason))
            console.print(
                Text("  DIFFER ", style="bold red")
                + Text(example.id.ljust(32) + " " + "judge said " + said + ", the label says " + example.label)
            )

    console.print("")
    console.print(
        Text(
            "Judge agreed with the human label on " + str(agreed) + " of " + str(len(labelled)) + ".",
            style="bold green" if agreed == len(labelled) else "bold yellow",
        )
    )
    if disagreements:
        console.print("")
        console.print("Where they differ:")
        for example, said, reason in disagreements:
            console.print("  " + example.id + ": judge said " + said + " because " + reason)
    console.print("")


@app.command()
def version() -> None:
    """Show the AgentQA version."""
    console.print("AgentQA " + __version__)


if __name__ == "__main__":
    app()
