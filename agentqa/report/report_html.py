"""Builds report.html out of one run's results.json and its traces.

Everything that came from an agent, a tool, a test file or a model is escaped
with html.escape before it reaches the page. A reply is not text we wrote: a
model can be talked into answering with `<script>`, and a ticket description in
the fake data already carries a planted instruction. The report has to be safe
to open even when the agent under test was not.

There is no templating library here on purpose. One file, the standard library,
and nothing to install before a person can read their own results.
"""

from __future__ import annotations

import json
from html import escape
from typing import Any

from agentqa.checks.check_result import (
    AREA_HONESTY,
    AREA_SAFETY,
    AREA_STEPS,
    AREA_TASK,
    AREAS_WORST_FIRST,
)
from agentqa.report.report_sentences import (
    AREA_TITLE,
    area_has_critical,
    card_note,
    counted,
    guard_note,
    release_sentences,
    runs_of,
    runs_total,
    tests_of,
)
from agentqa.report.report_styles import REPORT_CSS, REPORT_JS
from agentqa.repeat_runs import STABILITY_LABEL

PAGE_TITLE = "AgentQA test report"
DECISION_HEADING = "Release decision"

FILTERS = (("all", "All"), ("failed", "Failed"), ("flaky", "Flaky"), ("stable", "Stable"))

# The cards read in the order a person asks the questions, not worst first.
CARD_AREAS = (AREA_TASK, AREA_STEPS, AREA_SAFETY, AREA_HONESTY)

# What each trace event is called on the page.
EVENT_WORD = {
    "tool_call": "call",
    "tool_result": "result",
    "blocked": "blocked",
    "tool_error": "error",
    "final_reply": "reply",
    "step_limit_hit": "out of steps",
    "guard_blocked": "guard",
}

# What the two kinds of guard block are called on the page.
GUARD_STOPPED_LABEL = "Guard stopped"
GUARD_WRONGLY_BLOCKED_LABEL = "Guard wrongly blocked"


def safe(value: Any) -> str:
    """Every piece of text on the page goes through here. No exceptions."""
    if value is None:
        return ""
    return escape(str(value), quote=True)


def _short_json(value: Any, limit: int = 400) -> str:
    """A tool input or result, short enough to read in a list."""
    if value is None:
        return ""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def _thousands(number: Any) -> str:
    try:
        return format(int(number or 0), ",")
    except (TypeError, ValueError):
        return "0"


def _stability_of(test: dict[str, Any]) -> str:
    stability = test.get("stability")
    if stability:
        return str(stability)
    # A run saved before repeat runs existed has a plain status.
    status = str(test.get("status") or "")
    if status == "pass":
        return "stable_pass"
    if status == "error":
        return "error"
    return "stable_fail"


def _runs_text(test: dict[str, Any]) -> str:
    total = runs_total(test)
    passed = test.get("runs_passed")
    if passed is None:
        passed = total if _stability_of(test) == "stable_pass" else 0
    return str(passed) + "/" + str(total)


def _worst_failed_check(run: dict[str, Any]) -> dict[str, Any] | None:
    """The most serious failed check of one run: honesty, safety, task, steps."""
    failed = [check for check in (run.get("checks") or []) if not check.get("passed", True)]
    if not failed:
        return None
    order = {area: place for place, area in enumerate(AREAS_WORST_FIRST)}
    return min(failed, key=lambda check: order.get(str(check.get("area")), len(order)))


def _main_message(test: dict[str, Any]) -> str:
    """What happened, from the first run that did not pass."""
    for run in runs_of(test):
        if str(run.get("status")) == "pass":
            continue
        if run.get("error_message"):
            return str(run["error_message"])
        check = _worst_failed_check(run)
        if check is not None:
            return str(check.get("message") or "")
    return ""


def _failed_run_numbers(test: dict[str, Any]) -> list[int]:
    return [
        number
        for number, run in enumerate(runs_of(test), start=1)
        if str(run.get("status")) != "pass"
    ]


# ---------- the pieces of the page ----------


def _header(results: dict[str, Any]) -> str:
    run = results.get("run") or {}
    info = results.get("run_info") or {}
    summary = results.get("summary") or {}
    repeat = int(summary.get("repeat") or info.get("repeat") or 1)
    total = int(summary.get("total") or len(tests_of(results)))

    tests_pill = counted(total, "test")
    if repeat > 1:
        tests_pill += " × " + str(repeat) + " runs"

    model = info.get("model") or run.get("brain") or "unknown"
    brain = info.get("brain") or run.get("brain") or ""
    brain_pill = str(model) + ((" (" + str(brain) + ")") if brain and brain != model else "")

    agent_pill = str(info.get("agent") or "")
    framework = str(info.get("agent_framework") or "")
    if agent_pill and framework and framework != "none":
        agent_pill += " (" + framework + ")"

    pills = [
        str(run.get("started_at") or "")[:19].replace("T", " "),
        agent_pill,
        brain_pill,
        tests_pill,
        str(round(float(summary.get("total_time_s") or 0))) + " s",
    ]
    items = "\n".join("      <li class=\"pill\">" + safe(one) + "</li>" for one in pills if one)
    return (
        "    <div class=\"masthead\">\n"
        "      <h1>AgentQA</h1>\n"
        "      <span class=\"what\">Test report</span>\n"
        "    </div>\n"
        "    <ul class=\"pills\">\n" + items + "\n    </ul>\n"
    )


def _decision_box(results: dict[str, Any]) -> str:
    summary = results.get("summary") or {}
    release = str(summary.get("release") or "BLOCKED")
    ready = release == "READY"
    repeat = int(summary.get("repeat") or 1)
    counted_text = (
        str(summary.get("passed") or 0) + "/" + str(summary.get("total") or 0)
        + " passed" + (" every run" if repeat > 1 else "")
    )
    said = "\n".join(
        "      <p>" + safe(one) + "</p>" for one in release_sentences(results)
    )
    return (
        "    <h2>" + DECISION_HEADING + "</h2>\n"
        "    <div class=\"decision " + ("ready" if ready else "blocked") + "\">\n"
        "      <div class=\"verdict\">" + safe(release) + "</div>\n"
        "      <div class=\"counted\">" + safe(counted_text) + "</div>\n"
        + said + "\n"
        "    </div>\n"
    )


def _cards(results: dict[str, Any]) -> str:
    summary = results.get("summary") or {}
    areas = summary.get("areas") or {}
    cards = []
    for area in CARD_AREAS:
        count = areas.get(area) or {}
        passed = int(count.get("passed") or 0)
        total = int(count.get("total") or 0)
        good = passed == total
        tag = (
            "<span class=\"tag\">Critical</span>" if area_has_critical(results, area) else ""
        )
        note = card_note(results, area)
        if area == AREA_SAFETY:
            note = (note + " " + guard_note(results)).strip()
        cards.append(
            "      <div class=\"card " + ("good" if good else "bad") + "\">\n"
            "        <h3>" + safe(AREA_TITLE[area]) + tag + "</h3>\n"
            "        <div class=\"score\">" + safe(str(passed) + "/" + str(total)) + "</div>\n"
            "        <div class=\"note\">" + safe(note) + "</div>\n"
            "      </div>"
        )

    cards.append(
        "      <div class=\"card\">\n"
        "        <h3>Speed and cost</h3>\n"
        "        <div class=\"score\">"
        + safe(str(round(float(summary.get("total_time_s") or 0))) + " s")
        + "</div>\n"
        "        <div class=\"note\">"
        + safe(
            _thousands(summary.get("total_agent_tokens")) + " agent tokens, "
            + _thousands(summary.get("total_judge_tokens")) + " judge tokens."
        )
        + "</div>\n"
        "      </div>"
    )
    return "    <h2>The four areas</h2>\n    <div class=\"cards\">\n" + "\n".join(cards) + "\n    </div>\n"


def _all_tests_table(results: dict[str, Any]) -> str:
    buttons = "\n".join(
        "      <button type=\"button\" data-filter=\"" + key + "\" aria-pressed=\""
        + ("true" if key == "all" else "false") + "\">" + label + "</button>"
        for key, label in FILTERS
    )
    rows = []
    for test in tests_of(results):
        stability = _stability_of(test)
        rows.append(
            "          <tr data-stability=\"" + safe(stability) + "\">\n"
            "            <td><span class=\"word " + safe(stability) + "\">"
            + safe(STABILITY_LABEL.get(stability, stability)) + "</span></td>\n"
            "            <td>" + safe(test.get("id")) + "</td>\n"
            "            <td>" + safe(test.get("type")) + "</td>\n"
            "            <td>" + safe(_runs_text(test)) + "</td>\n"
            "            <td class=\"what\">" + safe(_main_message(test)) + "</td>\n"
            "          </tr>"
        )
    return (
        "    <h2>All tests</h2>\n"
        "    <div class=\"filters\">\n" + buttons + "\n    </div>\n"
        "    <div class=\"table-scroll\">\n"
        "      <table>\n"
        "        <thead>\n"
        "          <tr><th>Result</th><th>Test</th><th>Type</th><th>Runs</th>"
        "<th>What happened</th></tr>\n"
        "        </thead>\n"
        "        <tbody>\n" + "\n".join(rows) + "\n        </tbody>\n"
        "      </table>\n"
        "    </div>\n"
    )


def _wrongly_blocked_tools(run: dict[str, Any]) -> set[str]:
    """The tools this run's guard blocked although the test wanted them."""
    return {str(block.get("tool") or "") for block in (run.get("guard_wrongly_blocked") or [])}


def _guard_tag(tool: str, wrongly_blocked: set[str]) -> str:
    """Green when the guard stopped something, orange when it got in the way."""
    if tool in wrongly_blocked:
        return "<span class=\"guardtag wrong\">" + safe(GUARD_WRONGLY_BLOCKED_LABEL) + "</span> "
    return "<span class=\"guardtag stopped\">" + safe(GUARD_STOPPED_LABEL) + "</span> "


def _steps_list(trace: dict[str, Any], run: dict[str, Any] | None = None) -> str:
    events = (trace or {}).get("events") or []
    wrongly_blocked = _wrongly_blocked_tools(run or {})
    items = []
    for event in events:
        kind = str(event.get("kind") or "")
        if kind == "final_reply":
            continue
        word = EVENT_WORD.get(kind, kind)
        detail = event.get("error") or _short_json(event.get("output") or event.get("inputs"))
        tool = str(event.get("tool") or "")
        tag = _guard_tag(tool, wrongly_blocked) if kind == "guard_blocked" else ""
        items.append(
            "            <li class=\"" + safe(kind) + "\">"
            "<span class=\"kind\">" + safe(word) + "</span> "
            + tag
            + "<strong>" + safe(tool) + "</strong> "
            + safe(detail) + "</li>"
        )
    if not items:
        return "          <p class=\"asked\">The agent called no tools.</p>\n"
    return "          <ul class=\"steps\">\n" + "\n".join(items) + "\n          </ul>\n"


def _claim_box(test: dict[str, Any], run: dict[str, Any]) -> str:
    """Said it did X, really did Y. Only when the honesty check failed."""
    claim = None
    for check in run.get("checks") or []:
        if check.get("claimed_actions") is not None and not check.get("passed", True):
            claim = check
            break
    if claim is None:
        return ""

    said = ", ".join(str(one) for one in (claim.get("claimed_actions") or [])) or "nothing"
    trace_tools = []
    for event in ((run.get("trace") or {}).get("events") or []):
        if event.get("kind") == "tool_result" and event.get("tool"):
            if event["tool"] not in trace_tools:
                trace_tools.append(str(event["tool"]))
    really = ", ".join(trace_tools) + " only" if trace_tools else "nothing"
    tickets = len((run.get("ticket_system") or {}).get("new_ticket_ids") or [])
    return (
        "          <div class=\"claimbox\">\n"
        "            <strong>Claim vs reality.</strong> Said it did: " + safe(said)
        + ". Really did: " + safe(really)
        + ". New tickets in the system: " + safe(tickets) + ".\n"
        "          </div>\n"
    )


def _judge_reasons(run: dict[str, Any]) -> str:
    said = [
        check
        for check in (run.get("checks") or [])
        if check.get("source") == "judge" and check.get("reason")
    ]
    if not said:
        return ""
    lines = "".join(
        "<div>" + safe(check.get("check_name")) + ": " + safe(check.get("reason")) + "</div>"
        for check in said
    )
    return "          <div class=\"judge\"><strong>The judge said.</strong> " + lines + "</div>\n"


def _one_run(test: dict[str, Any], run: dict[str, Any], number: int, show_detail: bool) -> str:
    status = str(run.get("status") or "")
    word = {"pass": "passed", "fail": "failed", "error": "could not finish"}.get(status, status)
    head = (
        "        <div class=\"run\">\n"
        "          <h4>Run " + str(number) + " · "
        + "<span class=\"word " + ("stable_pass" if status == "pass" else "stable_fail") + "\">"
        + safe(word) + "</span></h4>\n"
    )
    if not show_detail:
        return head + "        </div>\n"

    body = _steps_list(run.get("trace") or {}, run)
    reply = run.get("final_reply")
    if reply:
        body += "          <div class=\"reply\">" + safe(reply) + "</div>\n"
    body += _claim_box(test, run)
    body += _judge_reasons(run)
    return head + body + "        </div>\n"


def _test_details(results: dict[str, Any]) -> str:
    blocks = []
    for test in tests_of(results):
        stability = _stability_of(test)
        open_now = stability != "stable_pass"
        failed_runs = set(_failed_run_numbers(test))
        runs = runs_of(test)
        bodies = "".join(
            _one_run(test, run, number, show_detail=(number in failed_runs))
            for number, run in enumerate(runs, start=1)
        )
        blocks.append(
            "    <details class=\"test\"" + (" open" if open_now else "") + ">\n"
            "      <summary><span class=\"word " + safe(stability) + "\">"
            + safe(STABILITY_LABEL.get(stability, stability)) + "</span> "
            + safe(test.get("id")) + " <span class=\"what\">" + safe(_runs_text(test))
            + "</span></summary>\n"
            "      <div class=\"body\">\n"
            "        <p class=\"asked\"><strong>The user asked:</strong> "
            + safe(test.get("user_asks")) + "</p>\n"
            + bodies
            + "      </div>\n"
            "    </details>"
        )
    return "    <h2>What happened in each test</h2>\n" + "\n".join(blocks) + "\n"


def _run_info_box(results: dict[str, Any]) -> str:
    info = results.get("run_info") or {}
    summary = results.get("summary") or {}
    rows = [
        ("Agent", str(info.get("agent") or "builtin")),
        ("Agent framework", info.get("agent_framework") or "none"),
        ("Guard prompt", info.get("guard_prompt_version") or "not used"),
        ("Brain", str(info.get("brain") or "") + " · " + str(info.get("model") or "")),
        ("Judge", str(info.get("judge_model") or info.get("judge_mode") or "words")),
        ("Judge prompt", info.get("judge_prompt_version") or "not used"),
        ("Claim prompt", info.get("claim_prompt_version") or "not used"),
        ("Agent rules hash", info.get("agent_rules_hash")),
        ("Test cases hash", info.get("test_cases_hash")),
        ("Repeats", summary.get("repeat") or info.get("repeat") or 1),
        ("AgentQA version", results.get("agentqa_version")),
    ]
    body = "\n".join(
        "          <tr><td>" + safe(name) + "</td><td>" + safe(value) + "</td></tr>"
        for name, value in rows
    )
    return (
        "    <h2>Run info</h2>\n"
        "    <div class=\"table-scroll\">\n"
        "      <table class=\"runinfo\">\n        <tbody>\n" + body
        + "\n        </tbody>\n      </table>\n    </div>\n"
    )


def build_report_html(results: dict[str, Any]) -> str:
    """One run as one self-contained page. Nothing is fetched when it opens."""
    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>" + safe(PAGE_TITLE) + "</title>\n"
        "<style>" + REPORT_CSS + "</style>\n"
        "</head>\n"
        "<body>\n"
        "  <div class=\"wrap\">\n"
        + _header(results)
        + _decision_box(results)
        + _cards(results)
        + _all_tests_table(results)
        + _test_details(results)
        + _run_info_box(results)
        + "    <footer>Made by AgentQA</footer>\n"
        "  </div>\n"
        "<script>" + REPORT_JS + "</script>\n"
        "</body>\n"
        "</html>\n"
    )
