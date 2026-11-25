"""The plain sentences on the report and the scorecard.

Every sentence here is written by code from the numbers in results.json. No
model writes them. A report that explains itself with a generated opinion is
not evidence, and a scorecard sentence has to be checkable against the table
right next to it.

The rule for the wording: say what failed, name the test, and give the run
count. Never soften it, never guess at a cause.
"""

from __future__ import annotations

from typing import Any

from agentqa.checks.check_result import (
    AREA_HONESTY,
    AREA_SAFETY,
    AREA_STEPS,
    AREA_TASK,
    AREAS_WORST_FIRST,
)

# What a failure in each area means, in words a person who did not write
# AgentQA can read.
AREA_TITLE = {
    AREA_TASK: "Did it finish the task?",
    AREA_STEPS: "Right steps?",
    AREA_SAFETY: "Was it safe?",
    AREA_HONESTY: "Was it honest?",
}

AREA_SHORT = {
    AREA_TASK: "Task",
    AREA_STEPS: "Steps",
    AREA_SAFETY: "Safety",
    AREA_HONESTY: "Honesty",
}

AREA_PROSE = {
    AREA_TASK: "did not finish the job",
    AREA_STEPS: "took the wrong steps",
    AREA_SAFETY: "did something it must never do",
    AREA_HONESTY: "said it had done something that never happened",
}

AREA_GOOD_NOTE = {
    AREA_TASK: "Every test finished the job.",
    AREA_STEPS: "The right tools, in the right order.",
    AREA_SAFETY: "No forbidden tool, no blocked call, nothing leaked.",
    AREA_HONESTY: "Every claim in a reply really happened.",
}

# Safety and Honesty failures are the ones that get the Critical tag.
CRITICAL_AREAS = (AREA_SAFETY, AREA_HONESTY)


def counted(number: int, thing: str) -> str:
    """Say '1 test', not '1 tests'."""
    return str(number) + " " + thing + ("" if number == 1 else "s")


def tests_of(results: dict[str, Any]) -> list[dict[str, Any]]:
    return list(results.get("tests") or [])


def runs_of(test: dict[str, Any]) -> list[dict[str, Any]]:
    """The runs of one test. A test saved before repeat runs has one run."""
    runs = test.get("runs")
    return list(runs) if runs else [test]


def runs_total(test: dict[str, Any]) -> int:
    return int(test.get("runs_total") or len(runs_of(test)) or 1)


def runs_failing_area(test: dict[str, Any], area: str) -> int:
    """How many runs of this test failed in that area."""
    return len([run for run in runs_of(test) if not (run.get("areas") or {}).get(area, True)])


def area_failures(results: dict[str, Any], area: str) -> list[tuple[str, int, int]]:
    """Per test that failed in this area: its id, failed runs and total runs."""
    failures = []
    for test in tests_of(results):
        if (test.get("areas") or {}).get(area, True):
            continue
        failures.append((str(test.get("id", "")), runs_failing_area(test, area), runs_total(test)))
    return failures


def _with_runs(test_id: str, failed: int, total: int, repeated: bool) -> str:
    if not repeated:
        return test_id
    return test_id + " in " + str(failed) + " of " + str(total) + " runs"


def _sentence_case(text: str) -> str:
    """Upper-case the first letter only. str.capitalize lowercases the rest."""
    return text[:1].upper() + text[1:] if text else text


def _joined(parts: list[str], limit: int = 3) -> str:
    """A list a person can read, cut short before it becomes a wall."""
    if len(parts) > limit:
        kept = parts[:limit]
        return ", ".join(kept) + ", and " + counted(len(parts) - limit, "more test")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def release_sentences(results: dict[str, Any]) -> list[str]:
    """One or two sentences saying why the release is READY or BLOCKED."""
    summary = results.get("summary") or {}
    repeat = int(summary.get("repeat") or 1)
    repeated = repeat > 1

    failing = [(area, area_failures(results, area)) for area in AREAS_WORST_FIRST]
    failing = [(area, found) for area, found in failing if found]

    if not failing:
        if int(summary.get("errors") or 0):
            return [
                "No test failed, but " + counted(int(summary["errors"]), "test")
                + " could not be run to the end, so the release is blocked."
            ]
        if repeated:
            return ["Every test passed in all " + str(repeat) + " runs."]
        return ["Every test passed."]

    worst_area, worst_found = failing[0]
    named = [_with_runs(test_id, failed, total, repeated) for test_id, failed, total in worst_found]
    first = (
        "In " + counted(len(worst_found), "test") + ", the agent "
        + AREA_PROSE[worst_area] + ": " + _joined(named) + "."
    )

    rest = []
    for area, found in failing[1:]:
        rest.append(AREA_SHORT[area] + " failed in " + counted(len(found), "test"))
    extra = []
    if rest:
        extra.append(_sentence_case(_joined(rest, limit=3)) + ".")
    counts = []
    if int(summary.get("flaky") or 0):
        counts.append(counted(int(summary["flaky"]), "test") + " passed in some runs and failed in others")
    if int(summary.get("errors") or 0):
        counts.append(counted(int(summary["errors"]), "test") + " could not be run to the end")
    if counts:
        extra.append(_sentence_case(_joined(counts, limit=2)) + ".")

    if not extra:
        return [first]
    return [first, " ".join(extra)]


def card_note(results: dict[str, Any], area: str) -> str:
    """The short note under one area card."""
    found = area_failures(results, area)
    if not found:
        return AREA_GOOD_NOTE[area]
    repeated = int((results.get("summary") or {}).get("repeat") or 1) > 1
    named = [_with_runs(test_id, failed, total, repeated) for test_id, failed, total in found]
    return "Failed: " + _joined(named, limit=2) + "."


def area_has_critical(results: dict[str, Any], area: str) -> bool:
    """Does this area carry a failure that counts as critical?"""
    return area in CRITICAL_AREAS and bool(area_failures(results, area))


def guard_note(results: dict[str, Any]) -> str:
    """What the guard did, in one short sentence. Empty when there was no guard.

    Both numbers are always shown. A guard that blocked something the user
    really wanted is as much a fact about the agent as one that stopped an
    attack, and a report that only printed the good number would be an advert.
    """
    info = results.get("run_info") or {}
    if not info.get("guard_prompt_version"):
        return ""
    summary = results.get("summary") or {}
    stopped = len(summary.get("guard_stopped") or [])
    wrong = len(summary.get("guard_wrongly_blocked") or [])
    return (
        "Guard stopped " + counted(stopped, "write")
        + ", wrongly blocked " + str(wrong) + "."
    )


def guard_sentence(results: dict[str, Any]) -> str:
    """The same facts, named, for the scorecard row. Empty without a guard."""
    note = guard_note(results)
    if not note:
        return ""
    summary = results.get("summary") or {}
    stopped = _guard_tools(summary.get("guard_stopped"))
    wrong = _guard_tools(summary.get("guard_wrongly_blocked"))
    said = "Guard: stopped " + (", ".join(stopped) if stopped else "nothing")
    said += ". Wrongly blocked " + (", ".join(wrong) if wrong else "nothing") + "."
    return said


def _guard_tools(blocks: Any) -> list[str]:
    """The tool names in a list of guard blocks, each named once, in order."""
    names: list[str] = []
    for block in blocks or []:
        name = str((block or {}).get("tool") or "")
        if name and name not in names:
            names.append(name)
    return names


def model_sentence(results: dict[str, Any]) -> str:
    """One factual sentence about one run, for the scorecard.

    "Safety: hidden-instruction failed 3 of 3 runs. Honesty: no-endless-loop
    failed 1 of 3 runs." Or, when nothing failed, one short sentence saying so.
    """
    parts = []
    for area in AREAS_WORST_FIRST:
        found = area_failures(results, area)
        if not found:
            continue
        named = [
            test_id + " failed " + str(failed) + " of " + str(total) + " runs"
            for test_id, failed, total in found
        ]
        parts.append(AREA_SHORT[area] + ": " + ", ".join(named) + ".")

    errors = int((results.get("summary") or {}).get("errors") or 0)
    if errors:
        parts.append("Errors: " + counted(errors, "test") + " could not be run to the end.")

    guard = guard_sentence(results)
    if guard:
        parts.append(guard)

    if not parts:
        return "Passed every test in every run."
    if parts == [guard]:
        return "Passed every test in every run. " + guard
    return " ".join(parts)
