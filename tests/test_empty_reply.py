"""An empty model answer must stay empty, and must be visible in the trace.

Writing words of our own into an empty reply hid a real model failure: the
judge then graded our sentence instead of the agent's answer.
"""

from __future__ import annotations

from fake_openai_client import FakeOpenAIClient, text_answer

from agentqa.agent_loop import run_agent_loop
from agentqa.brains.brain_interface import FinalReply, Message, ToolSpec
from agentqa.brains.brain_presets import BRAIN_LOCAL, preset_for
from agentqa.brains.openai_compatible_brain import OpenAICompatibleBrain
from agentqa.checks.reply_checks import run_reply_checks
from agentqa.fake_ticket_system.ticket_store import TicketStore
from agentqa.fake_ticket_system.tool_gateway import ToolGateway
from agentqa.trace import FINAL_REPLY, Trace

from conftest import make_test_case

PRESET = preset_for(BRAIN_LOCAL)


def make_brain(answers):
    client = FakeOpenAIClient(answers=answers)
    return OpenAICompatibleBrain(PRESET, model="test-model", client=client), client


def a_conversation():
    return [Message(role="system", content="rules"), Message(role="user", content="help me")]


class SpyJudge:
    """Counts every time it is asked, so a test can prove it was not."""

    def __init__(self) -> None:
        self.asked = []

    def judge(self, user_asks, reply, expectation):
        self.asked.append(reply)
        raise AssertionError("The judge must never be asked about an empty reply.")


# ---------- the brain ----------


def test_an_empty_answer_is_reported_as_empty_and_not_as_text():
    brain, _ = make_brain([text_answer("", finish_reason="length")])

    action = brain.next_action(a_conversation(), [])

    assert isinstance(action, FinalReply)
    assert action.reply == ""
    assert action.empty_reply is True
    assert action.finish_reason == "length"


def test_an_answer_that_is_only_a_think_block_is_empty():
    brain, _ = make_brain([text_answer("<think>I am still thinking</think>", finish_reason="length")])

    action = brain.next_action(a_conversation(), [])

    assert action.reply == ""
    assert action.empty_reply is True
    assert action.finish_reason == "length"


def test_an_answer_with_no_choices_at_all_is_empty():
    brain, _ = make_brain([text_answer("ignored")])
    brain.client.answers[0].choices = []

    action = brain.next_action(a_conversation(), [])

    assert action.reply == "" and action.empty_reply is True


def test_a_real_answer_keeps_its_finish_reason_and_is_not_empty():
    brain, _ = make_brain([text_answer("Here is your answer.", finish_reason="stop")])

    action = brain.next_action(a_conversation(), [])

    assert action.reply == "Here is your answer."
    assert action.empty_reply is False
    assert action.finish_reason == "stop"


# ---------- the trace ----------


class OneReplyBrain:
    """A brain that answers once, with whatever the test gave it."""

    def __init__(self, reply: FinalReply) -> None:
        self.reply = reply

    def next_action(self, conversation: list[Message], tools: list[ToolSpec]) -> FinalReply:
        return self.reply


def run_with(reply: FinalReply):
    case = make_test_case(max_steps=3)
    trace = Trace(test_id=case.id, acting_user=case.acting_user)
    gateway = ToolGateway(store=TicketStore.load_fresh(), acting_user=case.acting_user, trace=trace)
    return run_agent_loop(brain=OneReplyBrain(reply), gateway=gateway, test_case=case, tools=[])


def test_the_trace_records_an_empty_reply_as_empty():
    result = run_with(FinalReply(reply="", finish_reason="length", empty_reply=True))

    event = result.trace.events_of_kind(FINAL_REPLY)[-1]
    assert event.empty_reply is True
    assert event.finish_reason == "length"
    assert event.output == ""
    assert result.final_reply == ""


def test_the_trace_records_the_finish_reason_of_a_normal_reply():
    result = run_with(FinalReply(reply="All done.", finish_reason="stop"))

    event = result.trace.events_of_kind(FINAL_REPLY)[-1]
    assert event.empty_reply is False
    assert event.finish_reason == "stop"
    assert event.output == "All done."


# ---------- the checks ----------


def test_an_empty_reply_fails_the_reply_check_from_code():
    case = make_test_case(reply_should_mean="Says no.", reply_should_say=["no"])

    results = run_reply_checks(case, "", judge=SpyJudge())

    assert [r.check_name for r in results] == ["reply_given"]
    assert not results[0].passed
    assert results[0].message == "The agent gave no reply."


def test_an_empty_reply_is_never_sent_to_the_judge():
    case = make_test_case(reply_should_mean="Says no.")
    judge = SpyJudge()

    run_reply_checks(case, "   ", judge=judge)

    assert judge.asked == []


def test_a_real_reply_still_reaches_the_judge():
    case = make_test_case(reply_should_mean="Says no.")

    class YesJudge:
        def judge(self, user_asks, reply, expectation):
            from agentqa.judge.reply_judge import JudgeVerdict

            return JudgeVerdict(passed=True, reason="It says no.")

    results = run_reply_checks(case, "No, I may not.", judge=YesJudge())

    assert [r.check_name for r in results] == ["reply_given", "reply_means_the_right_thing"]
    assert all(r.passed for r in results)


# ---------- the thinking switch that caused the empty reply ----------


def test_the_local_preset_asks_for_no_thinking():
    """Ollama puts thinking in reasoning_content and leaves content empty.

    Its OpenAI endpoint ignores extra_body "think", so the request must carry
    reasoning_effort instead, or qwen3.5:4b answers with nothing at all.
    """
    brain, client = make_brain([text_answer("Done.")])

    brain.next_action(a_conversation(), [])

    assert client.requests[0]["reasoning_effort"] == "none"


def test_the_thinking_switch_is_dropped_once_if_the_provider_says_no():
    import httpx
    import openai

    refusal = openai.BadRequestError(
        "no such field",
        response=httpx.Response(400, request=httpx.Request("POST", "http://local/v1")),
        body=None,
    )
    brain, client = make_brain([refusal, text_answer("Done.")])

    action = brain.next_action(a_conversation(), [])

    assert action.reply == "Done."
    assert "reasoning_effort" not in client.requests[1]
