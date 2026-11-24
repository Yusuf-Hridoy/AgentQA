"""The agentqa command.

    agentqa run --brain fake
    agentqa run --brain local
    agentqa run --brain groq --only printer-ticket
    agentqa run --brain gemini --judge words
    agentqa run --brain groq --no-cache
    agentqa run --brain groq --repeat 3
    agentqa report reports/run-2026-11-24_12-08-47
    agentqa compare reports/run-A reports/run-B
    agentqa publish reports/run-A --name groq-repeat-3
    agentqa check-judge
    agentqa zoo

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
from agentqa.claims.claim_calibration_loader import ClaimExampleFileError, load_claim_examples
from agentqa.claims.claim_prompt import CLAIM_PROMPT_VERSION
from agentqa.claims.claim_reader import build_claim_reader
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
from agentqa.publish.publish_run import PublishError, publish_folder
from agentqa.release_decision import decide_release
from agentqa.repeat_runs import (
    MAX_REPEAT,
    MIN_REPEAT,
    STABLE_PASS,
    STABILITY_COLOUR,
    STABILITY_WORD,
    RepeatCountError,
    RepeatedTest,
    settled_repeat,
)
from agentqa.report.report_builder import ReportFolderError, write_report
from agentqa.results_writer import save_run
from agentqa.scorecard.scorecard_fairness import UnfairComparisonError
from agentqa.scorecard.scorecard_markdown import scorecard_markdown
from agentqa.scorecard.scorecard_writer import (
    SCORECARD_HTML_FILE_NAME,
    SCORECARD_MARKDOWN_FILE_NAME,
    save_scorecard,
)
from agentqa.run_all_tests import STATUS_ERROR, STATUS_PASS, TestOutcome, run_one_test
from agentqa.run_info import build_run_info
from agentqa.test_case_loader import TestCaseFileError, load_test_cases
from agentqa.zoo.zoo_loader import ZooFileError
from agentqa.zoo.zoo_results_writer import ZOO_RESULTS_FILE_NAME, save_zoo_run
from agentqa.zoo.zoo_runner import CAUGHT, ZooResult, run_zoo, zoo_exit_code

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


def _times() -> str:
    """'×' when the terminal speaks UTF-8, else a plain x."""
    return "×" if _dot() == "·" else "x"


def _short_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _thousands(number: int) -> str:
    return format(int(number), ",")


def _counted(number: int, thing: str) -> str:
    """Say '1 test', not '1 tests'."""
    return str(number) + " " + thing + ("" if number == 1 else "s")


def _print_one_result(outcome: TestOutcome) -> None:
    """The word is always printed, not only the colour."""
    word = STATUS_WORD.get(outcome.status, "FAIL")
    colour = STATUS_COLOUR.get(outcome.status, "red")
    line = Text("  ")
    line.append(word.ljust(6), style="bold " + colour)
    line.append(" " + outcome.test_case.id.ljust(24))
    if outcome.status != STATUS_PASS:
        extra = outcome.first_failure_message
        worst = outcome.worst_failed_check
        if worst is not None:
            extra = worst.area.upper() + " " + _dot() + " " + extra
        line.append(extra, style=colour)
    console.print(line)


def _print_one_repeat_result(repeated: RepeatedTest) -> None:
    """One line per test when a test ran more than once.

    PASS, FLAKY, FAIL or ERROR, then how many runs passed, then the worst thing
    that went wrong in the first run that did not pass.
    """
    stability = repeated.stability
    word = STABILITY_WORD[stability]
    colour = STABILITY_COLOUR[stability]

    line = Text("  ")
    line.append(word.ljust(6), style="bold " + colour)
    line.append("  " + repeated.test_case.id.ljust(24))
    line.append(repeated.runs_text.ljust(6))
    if stability != STABLE_PASS:
        said = repeated.main_message
        area = repeated.main_area
        number = repeated.first_failed_run_number
        if area:
            said = area.upper() + " " + _dot() + " run " + str(number) + ": " + said
        elif number:
            said = "run " + str(number) + ": " + said
        line.append(said, style=colour)
    console.print(line)


def _action_list(actions: list[str]) -> str:
    """A list of action names a person can read. Empty means no claim."""
    return ", ".join(actions) if actions else "nothing"


def _check_claim_reader(reader: object, examples: list) -> tuple[int, list[tuple[str, list[str], list[str]]]]:
    """Run the AI claim reader on the labelled examples. Prints one line each."""
    matched = 0
    differences: list[tuple[str, list[str], list[str]]] = []
    for example in examples:
        try:
            read = list(reader.read_claims(example.user_asks, example.reply))
        except JudgeError as problem:
            console.print(Text("  ERROR  " + example.id.ljust(32) + " " + str(problem), style="yellow"))
            differences.append((example.id, list(example.expected), []))
            continue

        if read == list(example.expected):
            matched += 1
            console.print(
                Text("  MATCH  ", style="bold green")
                + Text(example.id.ljust(32) + " " + _action_list(read))
            )
        else:
            differences.append((example.id, list(example.expected), read))
            console.print(
                Text("  DIFFER ", style="bold red")
                + Text(
                    example.id.ljust(32) + " read " + _action_list(read)
                    + ", the label says " + _action_list(example.expected)
                )
            )
    return matched, differences


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
    repeat: int = typer.Option(
        1,
        "--repeat",
        help="Run every test this many times, from " + str(MIN_REPEAT) + " to " + str(MAX_REPEAT)
        + ". One run cannot show whether an agent is reliable.",
    ),
) -> None:
    """Run the AgentQA test cases against an agent."""
    started_at = datetime.now()

    try:
        times = settled_repeat(repeat)
        test_cases = load_test_cases(cases, only=only)
    except (RepeatCountError, TestCaseFileError) as problem:
        _stop_with_setup_error(problem)

    # Everything that can fail before the first test, fails before the first test.
    try:
        judge_mode = settled_judge_mode(judge, brain)
        check_brain_ready(brain, [case.id for case in test_cases])
        reply_judge = build_judge(judge_mode, use_cache=not no_cache)
        claim_reader = build_claim_reader(judge_mode, use_cache=not no_cache)
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
            "AgentQA " + dot + " " + _counted(len(test_cases), "test")
            + ((" " + _times() + " " + str(times) + " runs") if times > 1 else "") + " "
            + dot + " brain: " + _header_brain(brain)
            + " " + dot + " judge: " + (judge_model or "words"),
            style="bold",
        )
    )
    console.print("")

    outcomes = []
    for test_case in test_cases:
        # Every run gets a fresh brain and fresh fake data, as one run does.
        runs = [
            run_one_test(
                test_case,
                brain_name=brain,
                judge=reply_judge,
                claim_reader=claim_reader,
            )
            for _ in range(times)
        ]
        repeated = RepeatedTest(test_case=test_case, runs=runs)
        outcomes.append(repeated)
        if times > 1:
            _print_one_repeat_result(repeated)
        else:
            _print_one_result(runs[0])

    decision = decide_release(outcomes, repeat=times)
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
            claim_prompt_version=CLAIM_PROMPT_VERSION if claim_reader is not None else None,
            repeat=times,
        ),
    )
    report_file = write_report(run_folder)

    areas = decision.areas
    console.print("")
    console.print(
        "Task: " + areas["task"].as_text()
        + "   Steps: " + areas["steps"].as_text()
        + "   Safety: " + areas["safety"].as_text()
        + "   Honesty: " + areas["honesty"].as_text()
        + "   Critical failures: " + str(decision.critical_failures)
        + ("   Flaky: " + str(decision.flaky) if times > 1 else "")
        + "   Errors: " + str(decision.errors),
        soft_wrap=True,
    )
    console.print(
        "Agent tokens: " + _thousands(decision.total_agent_tokens)
        + " " + dot + " Judge tokens: " + _thousands(decision.total_judge_tokens)
        + " " + dot + " Time: " + str(round(decision.total_time_s)) + " s"
    )
    result_style = "bold green" if decision.is_ready else "bold red"
    console.print(
        Text(
            "Result: " + decision.passed_text() + " " + dot + " Release: " + decision.status,
            style=result_style,
        )
    )
    console.print("Report: " + _short_path(report_file))
    console.print("")

    raise typer.Exit(code=decision.exit_code())


@app.command()
def report(
    run_folder: Path = typer.Argument(..., help="A run folder that agentqa run made."),
) -> None:
    """Build report.html again from a saved run folder."""
    try:
        path = write_report(run_folder)
    except ReportFolderError as problem:
        _stop_with_setup_error(problem)
        return

    console.print("")
    console.print("Report: " + _short_path(path))
    console.print("")


@app.command()
def compare(
    run_folders: list[Path] = typer.Argument(..., help="Two or more run folders to compare."),
    reports_dir: Path = typer.Option(None, "--reports-dir", help="Where to save the scorecard folder."),
) -> None:
    """Put several runs in one table, but only when they can be compared fairly."""
    try:
        folder, rows, facts = save_scorecard(list(run_folders), reports_dir=reports_dir)
    except (ReportFolderError, UnfairComparisonError) as problem:
        _stop_with_setup_error(problem)
        return

    console.print("")
    # Plain text, not markup: a table must not be re-wrapped or re-read as style.
    console.print(Text(scorecard_markdown(rows, facts)), soft_wrap=True)
    console.print("Scorecard: " + _short_path(folder / SCORECARD_HTML_FILE_NAME))
    console.print("Markdown:  " + _short_path(folder / SCORECARD_MARKDOWN_FILE_NAME))
    console.print("")


@app.command()
def publish(
    source_folder: Path = typer.Argument(..., help="A run folder or a scorecard folder."),
    name: str = typer.Option(None, "--name", help="What to call it in docs/reports/."),
    docs_dir: Path = typer.Option(None, "--docs-dir", help="Where the site lives. Normally docs/."),
) -> None:
    """Copy a run into docs/, so anyone can open it on the web."""
    try:
        target, copied, index = publish_folder(source_folder, name=name, docs_dir=docs_dir)
    except PublishError as problem:
        _stop_with_setup_error(problem)
        return

    console.print("")
    console.print(
        "Published " + _counted(len(copied), "file") + " to " + _short_path(target)
    )
    console.print("Landing page: " + _short_path(index))
    console.print("")


@app.command("check-judge")
def check_judge(
    examples: Path = typer.Option(None, "--examples", help="The judge example file to use."),
    claims: Path = typer.Option(None, "--claims", help="The claim example file to use."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Do not reuse saved judge answers."),
) -> None:
    """Measure the judge and the claim reader against hand-labelled examples."""
    try:
        labelled = load_judge_examples(examples)
        claim_examples = load_claim_examples(claims)
        reply_judge = build_judge(JUDGE_GEMINI, use_cache=not no_cache)
        claim_reader = build_claim_reader(JUDGE_GEMINI, use_cache=not no_cache)
    except (JudgeExampleFileError, ClaimExampleFileError, BrainSetupError) as problem:
        _stop_with_setup_error(problem)

    dot = _dot()
    console.print("")
    console.print(
        Text(
            "AgentQA check-judge " + dot + " " + _counted(len(labelled), "judge example")
            + " " + dot + " " + _counted(len(claim_examples), "claim example")
            + " " + dot + " " + reply_judge.model
            + " " + dot + " " + JUDGE_PROMPT_VERSION + " " + dot + " " + CLAIM_PROMPT_VERSION,
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
    matched, claim_differences = _check_claim_reader(claim_reader, claim_examples)

    console.print("")
    console.print(
        Text(
            "Judge agreed with the human label on " + str(agreed) + " of " + str(len(labelled)) + ".",
            style="bold green" if agreed == len(labelled) else "bold yellow",
        )
    )
    console.print(
        Text(
            "Claim reader matched the human label on " + str(matched)
            + " of " + str(len(claim_examples)) + ".",
            style="bold green" if matched == len(claim_examples) else "bold yellow",
        )
    )
    if disagreements:
        console.print("")
        console.print("Where the judge differs:")
        for example, said, reason in disagreements:
            console.print("  " + example.id + ": judge said " + said + " because " + reason)
    if claim_differences:
        console.print("")
        console.print("Where the claim reader differs:")
        for example_id, expected, actual in claim_differences:
            console.print(
                "  " + example_id + ": expected " + _action_list(expected)
                + ", read " + _action_list(actual)
            )
    console.print("")


def _print_one_zoo_result(result: ZooResult, width: int = 9) -> None:
    """One line per broken agent. The word is always printed, not only the colour."""
    colour = "green" if result.is_caught else "red"
    line = Text("  ")
    line.append(result.result.ljust(width), style="bold " + colour)
    line.append("  ")
    line.append(result.agent.name.ljust(18))
    if result.is_caught:
        line.append("by " + result.agent.expected_check + ": " + result.catching_message)
    elif result.result == CAUGHT:
        line.append(result.catching_message, style=colour)
    else:
        line.append(
            _why_not_caught(result),
            style=colour,
        )
    console.print(line)


def _why_not_caught(result: ZooResult) -> str:
    """Plain words for a bug that got through, or that the wrong check found."""
    expected = result.agent.expected_check
    if not result.failed_checks:
        return "no check failed, so " + expected + " did not notice the bug"
    return (
        expected + " passed. Failed instead: " + ", ".join(result.failed_checks)
    )


@app.command()
def zoo(
    only: str = typer.Option(None, "--only", help="Run one broken agent by its name."),
    zoo_dir: Path = typer.Option(None, "--zoo-dir", help="Folder with the broken agents."),
    cases: Path = typer.Option(TEST_CASES_DIR, "--cases", help="Folder with the test case files."),
    reports_dir: Path = typer.Option(None, "--reports-dir", help="Where to save the zoo folder."),
) -> None:
    """Run the broken agents. Every bug must be caught by the right check."""
    started_at = datetime.now()
    dot = _dot()

    try:
        results = run_zoo(zoo_folder=zoo_dir, cases_folder=cases, only=only)
    except (ZooFileError, TestCaseFileError, FakeBrainScriptError) as problem:
        _stop_with_setup_error(problem)
        return

    console.print("")
    console.print(
        Text(
            "AgentQA bug zoo " + dot + " " + _counted(len(results), "broken agent")
            + " " + dot + " each must be caught",
            style="bold",
        )
    )
    console.print("")
    width = max(len(result.result) for result in results)
    for result in results:
        _print_one_zoo_result(result, width=width)

    finished_at = datetime.now()
    zoo_folder = save_zoo_run(results, started_at, finished_at, reports_dir=reports_dir)

    caught = [one for one in results if one.is_caught]
    code = zoo_exit_code(results)
    missed = [one.agent.name for one in results if not one.is_caught]
    console.print("")
    console.print(
        Text(
            "Caught " + str(len(caught)) + " of " + str(len(results)) + " " + dot + " "
            + ("no blind spots" if code == 0 else "blind spots: " + ", ".join(missed)),
            style="bold green" if code == 0 else "bold red",
        )
    )
    console.print("Report: " + _short_path(zoo_folder / ZOO_RESULTS_FILE_NAME))
    console.print("")

    raise typer.Exit(code=code)


@app.command()
def version() -> None:
    """Show the AgentQA version."""
    console.print("AgentQA " + __version__)


if __name__ == "__main__":
    app()
