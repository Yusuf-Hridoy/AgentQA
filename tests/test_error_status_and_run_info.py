"""Errors, least privilege, run info and the release decision.

A failing API call must cost one test, never the run. A run must say what it
was made of. And the brain must only see the tools the acting user may use.
"""

from __future__ import annotations

import json
from datetime import datetime

from fake_openai_client import FakeOpenAIClient, text_answer, tool_answer

from agentqa.agent_loop import AGENT_RULES, starting_conversation, tools_the_user_may_use
from agentqa.brains.brain_errors import BrainError
from agentqa.brains.brain_interface import FinalReply, ToolSpec
from agentqa.brains.brain_presets import BRAIN_GROQ, preset_for
from agentqa.brains.fake_brain import FakeBrain
from agentqa.brains.openai_compatible_brain import OpenAICompatibleBrain
from agentqa.judge.reply_judge import JudgeError
from agentqa.project_paths import TEST_CASES_DIR
from agentqa.release_decision import BLOCKED, READY, decide_release
from agentqa.results_writer import RESULTS_FILE_NAME, save_run
from agentqa.run_all_tests import STATUS_ERROR, STATUS_FAIL, STATUS_PASS, run_all_tests, run_one_test
# test_cases_hash is renamed here, or pytest would try to collect it as a test.
from agentqa.run_info import agent_rules_hash, build_run_info, short_hash
from agentqa.run_info import test_cases_hash as hash_of_test_cases
from agentqa.test_case_loader import load_test_cases

PRESET = preset_for(BRAIN_GROQ)


def case_called(test_id):
    return [case for case in load_test_cases(TEST_CASES_DIR) if case.id == test_id][0]


class JudgeThatAlwaysFails:
    """Stands in for a judge whose provider is down."""

    def judge(self, user_request, reply, expectation):
        raise JudgeError("The judge did not answer.")

    def usage_totals(self):
        from agentqa.brains.brain_interface import TokenUsage

        return TokenUsage()


# ---------- one error does not stop the run ----------


def test_a_brain_error_marks_only_that_test_as_error():
    case = case_called("printer-ticket")
    client = FakeOpenAIClient(
        answers=[
            tool_answer([("call_1", "search_help_docs", '{"query": "printer"}')]),
            BrainError("Groq did not answer after 3 tries."),
        ]
    )
    brain = OpenAICompatibleBrain(PRESET, model="test-model", client=client)

    outcome = run_one_test(case, brain=brain)

    assert outcome.status == STATUS_ERROR
    assert outcome.is_error
    assert not outcome.passed
    assert outcome.error_message == "Groq did not answer after 3 tries."
    assert outcome.checks == []
    # The partial trace is kept.
    assert outcome.trace.called_tool_names_in_order() == ["search_help_docs"]


def test_the_next_test_still_runs_after_an_error():
    cases = [case_called("printer-ticket"), case_called("check-ticket-status")]
    outcomes = []
    for number, case in enumerate(cases):
        if number == 0:
            client = FakeOpenAIClient(answers=[BrainError("Groq did not answer after 3 tries.")])
            brain = OpenAICompatibleBrain(PRESET, model="test-model", client=client)
        else:
            brain = FakeBrain.for_test_case(case.id)
        outcomes.append(run_one_test(case, brain=brain))

    assert [outcome.status for outcome in outcomes] == [STATUS_ERROR, STATUS_PASS]


def test_a_judge_error_marks_the_test_as_error():
    case = case_called("printer-ticket")

    outcome = run_one_test(case, brain=FakeBrain.for_test_case(case.id), judge=JudgeThatAlwaysFails())

    assert outcome.status == STATUS_ERROR
    assert "judge" in outcome.error_message.lower()


def test_a_critical_test_that_errors_is_not_counted_as_a_critical_failure():
    case = case_called("hidden-instruction")
    client = FakeOpenAIClient(answers=[BrainError("Groq did not answer after 3 tries.")])
    brain = OpenAICompatibleBrain(PRESET, model="test-model", client=client)

    outcome = run_one_test(case, brain=brain)

    assert outcome.is_error
    assert not outcome.is_critical_failure


# ---------- the release decision ----------


def test_one_error_blocks_the_release():
    good = run_one_test(case_called("printer-ticket"), brain_name="fake")
    client = FakeOpenAIClient(answers=[BrainError("Groq did not answer after 3 tries.")])
    broken = run_one_test(
        case_called("check-ticket-status"),
        brain=OpenAICompatibleBrain(PRESET, model="test-model", client=client),
    )

    decision = decide_release([good, broken])

    assert decision.status == BLOCKED
    assert decision.errors == 1
    assert decision.error_test_ids == ["check-ticket-status"]
    assert decision.passed_tests == 1
    assert decision.exit_code() == 1


def test_errors_and_failures_are_counted_apart():
    good = run_one_test(case_called("printer-ticket"), brain_name="fake")
    bad = run_one_test(case_called("printer-ticket"), brain=FakeBrain([FinalReply(reply="Nope.")]))
    client = FakeOpenAIClient(answers=[BrainError("down")])
    broken = run_one_test(
        case_called("check-ticket-status"),
        brain=OpenAICompatibleBrain(PRESET, model="test-model", client=client),
    )

    decision = decide_release([good, bad, broken])

    assert bad.status == STATUS_FAIL
    assert decision.errors == 1
    assert decision.failed_test_ids == ["printer-ticket"]
    assert decision.to_dict()["errors"] == 1


def test_a_run_with_no_errors_and_no_failures_is_ready():
    outcomes = run_all_tests(load_test_cases(TEST_CASES_DIR), brain_name="fake")
    decision = decide_release(outcomes)

    assert decision.status == READY
    assert decision.errors == 0
    assert decision.passed_tests == 10


# ---------- tokens and time ----------


def test_tokens_and_time_are_recorded_per_test():
    case = case_called("printer-ticket")
    client = FakeOpenAIClient(
        answers=[
            tool_answer([("call_1", "search_help_docs", '{"query": "printer"}')], 100, 20),
            tool_answer(
                [("call_2", "create_ticket", '{"title": "Printer", "description": "Not printing"}')],
                120,
                25,
            ),
            text_answer("I made a ticket for your printer problem.", 140, 30),
        ]
    )
    brain = OpenAICompatibleBrain(PRESET, model="test-model", client=client)

    outcome = run_one_test(case, brain=brain)
    decision = decide_release([outcome])

    assert outcome.agent_tokens.total_tokens == 360 + 75
    assert outcome.judge_tokens.total_tokens == 0
    assert outcome.duration_ms > 0
    assert decision.total_agent_tokens == 435


# ---------- Fix A: least privilege ----------


def test_an_employee_only_sees_the_employee_tools():
    names = [spec.name for spec in tools_the_user_may_use("employee")]

    assert names == ["search_help_docs", "get_ticket", "create_ticket", "update_ticket"]
    assert "assign_ticket" not in names
    assert "send_notification" not in names


def test_an_it_agent_sees_all_six_tools():
    names = [spec.name for spec in tools_the_user_may_use("it_agent")]

    assert len(names) == 6
    assert "assign_ticket" in names and "send_notification" in names


def test_the_brain_is_told_who_it_helps_and_which_role():
    case = case_called("printer-ticket")

    conversation = starting_conversation(case, "employee")

    assert conversation[0].content.startswith(AGENT_RULES)
    assert "You are helping alice (role: employee)." in conversation[0].content


def test_the_running_agent_only_sees_the_tools_its_user_may_use():
    seen: list[list[str]] = []

    class WatchingBrain:
        def next_action(self, conversation, tools: list[ToolSpec]):
            seen.append([spec.name for spec in tools])
            return FinalReply(reply="Nothing to do.")

    run_one_test(case_called("printer-ticket"), brain=WatchingBrain())  # alice, an employee
    run_one_test(case_called("notify-needs-consent"), brain=WatchingBrain())  # maria, an IT agent

    assert "assign_ticket" not in seen[0]
    assert len(seen[0]) == 4
    assert len(seen[1]) == 6


# ---------- run info and hashes ----------


def test_the_run_info_says_what_the_run_was_made_of():
    info = build_run_info(
        brain_name="groq",
        provider="groq",
        model="openai/gpt-oss-120b",
        judge_mode="gemini",
        judge_model="gemini-3.5-flash-lite",
        judge_prompt_version="judge-v1",
        cases_folder=TEST_CASES_DIR,
    )

    assert info["brain"] == "groq"
    assert info["provider"] == "groq"
    assert info["model"] == "openai/gpt-oss-120b"
    assert info["judge_mode"] == "gemini"
    assert info["judge_prompt_version"] == "judge-v1"
    assert len(info["agent_rules_hash"]) == 8
    assert len(info["test_cases_hash"]) == 8
    assert info["agentqa_version"]


def test_the_agent_rules_hash_changes_when_the_rules_change():
    assert agent_rules_hash() == short_hash(AGENT_RULES)
    assert agent_rules_hash() != short_hash(AGENT_RULES + " One more rule.")


def test_the_test_cases_hash_changes_when_a_test_case_file_changes(tmp_path):
    folder = tmp_path / "cases"
    folder.mkdir()
    one = folder / "a.yaml"
    one.write_text("id: a\n", encoding="utf-8")
    before = hash_of_test_cases(folder)

    one.write_text("id: a\ntype: normal\n", encoding="utf-8")
    after = hash_of_test_cases(folder)

    assert before != after
    assert hash_of_test_cases(folder) == after


def test_results_json_holds_the_run_info_status_tokens_and_time(tmp_path):
    outcomes = run_all_tests(load_test_cases(TEST_CASES_DIR), brain_name="fake")
    now = datetime.now()

    folder = save_run(
        outcomes=outcomes,
        decision=decide_release(outcomes),
        brain_name="fake",
        started_at=now,
        finished_at=now,
        cases_folder="test_cases",
        reports_dir=tmp_path,
        run_info=build_run_info("fake", None, "fake", "words", None, None, TEST_CASES_DIR),
    )
    results = json.loads((folder / RESULTS_FILE_NAME).read_text(encoding="utf-8"))

    assert results["run_info"]["brain"] == "fake"
    assert results["run_info"]["judge_mode"] == "words"
    assert len(results["run_info"]["agent_rules_hash"]) == 8
    assert results["summary"]["errors"] == 0
    assert "total_agent_tokens" in results["summary"]
    assert "total_time_s" in results["summary"]
    for test in results["tests"]:
        assert test["status"] in (STATUS_PASS, STATUS_FAIL, STATUS_ERROR)
        assert "agent_tokens" in test and "judge_tokens" in test
        assert "duration_ms" in test


def test_no_key_is_written_into_the_results(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "sk-should-never-be-written")
    outcomes = run_all_tests(load_test_cases(TEST_CASES_DIR), brain_name="fake")
    now = datetime.now()

    folder = save_run(
        outcomes=outcomes,
        decision=decide_release(outcomes),
        brain_name="fake",
        started_at=now,
        finished_at=now,
        reports_dir=tmp_path,
        run_info=build_run_info("fake", None, "fake", "words", None, None, TEST_CASES_DIR),
    )
    written = "\n".join(path.read_text(encoding="utf-8") for path in folder.rglob("*.json"))

    assert "sk-should-never-be-written" not in written


# ---------- an error is not a pass in any area ----------


def nine_passes_and_one_error():
    """Nine real passing tests, plus one that errored before any check ran."""
    cases = load_test_cases(TEST_CASES_DIR)
    outcomes = [run_one_test(case, brain_name="fake") for case in cases[:9]]
    client = FakeOpenAIClient(answers=[BrainError("Gemini hit its rate limit after 3 tries.")])
    outcomes.append(
        run_one_test(cases[9], brain=OpenAICompatibleBrain(PRESET, model="test-model", client=client))
    )
    return outcomes


def test_an_errored_test_passes_no_area():
    case = case_called("printer-ticket")
    client = FakeOpenAIClient(answers=[BrainError("down")])

    outcome = run_one_test(case, brain=OpenAICompatibleBrain(PRESET, model="test-model", client=client))

    assert outcome.is_error
    assert outcome.area_passed() == {"task": False, "steps": False, "safety": False}


def test_nine_passes_and_one_error_score_nine_of_ten_in_every_area():
    outcomes = nine_passes_and_one_error()

    decision = decide_release(outcomes)

    assert [outcome.status for outcome in outcomes].count(STATUS_PASS) == 9
    assert decision.errors == 1
    assert decision.areas["task"].as_text() == "9/10"
    assert decision.areas["steps"].as_text() == "9/10"
    assert decision.areas["safety"].as_text() == "9/10"
    assert decision.status == BLOCKED


def test_an_error_never_flatters_the_safety_score():
    """Not knowing whether the agent was safe is not the same as it being safe."""
    outcomes = nine_passes_and_one_error()

    safety = decide_release(outcomes).areas["safety"]

    assert safety.passed == 9
    assert safety.total == 10


# ---------- the agent rules ----------


def test_the_rules_tell_the_agent_to_search_the_help_docs_first():
    """A deliberate rule change. The hash below is what makes runs comparable."""
    assert "search the help docs for a fix" in AGENT_RULES
    assert "Before making a new ticket" in AGENT_RULES


def test_the_agent_is_shown_the_rules_it_is_judged_against():
    conversation = starting_conversation(case_called("printer-ticket"), "employee")

    assert "Before making a new ticket, search the help docs" in conversation[0].content
