"""Is it fair to put these runs in one table?

A scorecard that compares runs made under different rules is worse than no
scorecard: it looks like evidence and is not. If one run was judged by a
different judge prompt, or ran against an edited test case, the numbers are not
about the models any more.

So `agentqa compare` refuses, and says which value differs and in which run.
There is no override flag on purpose. Re-run the models instead.
"""

from __future__ import annotations

from typing import Any

# What must match across every run, and what to call it in the message.
MUST_MATCH = (
    ("agent_rules_hash", "agent rules"),
    ("test_cases_hash", "test cases"),
    ("judge_prompt_version", "judge prompt"),
    ("claim_prompt_version", "claim prompt"),
)

# The repeat count may differ. More runs of one model is more evidence about
# that model, not a different question being asked.
#
# The agent and the model may differ too, and that is the point of Phase 5: a
# scorecard that could not hold a builtin run next to a LangGraph run would
# answer nothing. What may never differ is the rules the agents were held to.
MAY_DIFFER = ("repeat", "agent", "agent_framework", "brain", "model")

# The guard is part of the agent under test, so two agents with different
# guards are two different agents, which is allowed. Two guard runs with
# different guard prompts are not: the guard would have been told different
# things, and the numbers would be about the prompt, not about the agent.
GUARD_PROMPT_FIELD = "guard_prompt_version"


class UnfairComparisonError(Exception):
    """Two runs were not made under the same rules, so they cannot be compared."""


def value_of(results: dict[str, Any], field_name: str) -> Any:
    return (results.get("run_info") or {}).get(field_name)


def check_fair(runs: list[tuple[str, dict[str, Any]]]) -> None:
    """Raise when the runs were not made under the same rules.

    `runs` is a list of (name a person recognises, results.json as a dict).
    """
    if len(runs) < 2:
        raise UnfairComparisonError("Give at least two run folders to compare.")

    first_name, first = runs[0]
    for field_name, plain_name in MUST_MATCH:
        wanted = value_of(first, field_name)
        for name, results in runs[1:]:
            found = value_of(results, field_name)
            if found == wanted:
                continue
            raise UnfairComparisonError(
                "These runs used different " + plain_name + " ("
                + _said(wanted) + " vs " + _said(found) + "), so they can't be "
                "compared fairly. " + _said(wanted) + " is from " + first_name
                + " and " + _said(found) + " is from " + name + "."
            )

    check_guard_prompts_match(runs)


def check_guard_prompts_match(runs: list[tuple[str, dict[str, Any]]]) -> None:
    """Two runs that both had a guard must have had the same guard prompt."""
    guarded = [
        (name, value_of(results, GUARD_PROMPT_FIELD))
        for name, results in runs
        if value_of(results, GUARD_PROMPT_FIELD)
    ]
    if len(guarded) < 2:
        return

    first_name, wanted = guarded[0]
    for name, found in guarded[1:]:
        if found == wanted:
            continue
        raise UnfairComparisonError(
            "These runs used different guard prompts (" + _said(wanted) + " vs "
            + _said(found) + "), so they can't be compared fairly. "
            + _said(wanted) + " is from " + first_name + " and " + _said(found)
            + " is from " + name + "."
        )


def _said(value: Any) -> str:
    """A value a person can read, even when it was never recorded."""
    if value is None or value == "":
        return "none recorded"
    return str(value)


def shared_facts(runs: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    """The values every run shares. Only call this once check_fair has passed."""
    _name, first = runs[0]
    return {field_name: value_of(first, field_name) for field_name, _plain in MUST_MATCH}
