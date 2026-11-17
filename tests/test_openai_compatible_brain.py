"""The real brain, with a fake client. No network, no key, no Ollama."""

from __future__ import annotations

import httpx
import openai
import pytest
from fake_openai_client import FakeFunction, FakeOpenAIClient, text_answer, tool_answer

from agentqa.brains.brain_errors import BrainError
from agentqa.brains.brain_interface import FinalReply, Message, ToolCall
from agentqa.brains.brain_presets import BRAIN_GROQ, preset_for
from agentqa.brains.openai_compatible_brain import (
    RAW_INPUT_KEY,
    OpenAICompatibleBrain,
    clean_schema,
    conversation_as_openai,
    tool_call_from_model,
    strip_thinking,
    tools_as_openai,
)
from agentqa.fake_ticket_system.it_support_tools import tool_specs

PRESET = preset_for(BRAIN_GROQ)
TOOLS = tool_specs(["search_help_docs", "create_ticket"])


def make_brain(answers):
    client = FakeOpenAIClient(answers=answers)
    brain = OpenAICompatibleBrain(PRESET, model="test-model", client=client)
    return brain, client


def a_conversation():
    return [Message(role="system", content="rules"), Message(role="user", content="help me")]


# ---------- tool calls ----------


def test_a_tool_call_comes_back_with_its_id_and_inputs():
    brain, _ = make_brain([tool_answer([("call_7", "create_ticket", '{"title": "T", "description": "D"}')])])

    action = brain.next_action(a_conversation(), TOOLS)

    assert isinstance(action, ToolCall)
    assert action.tool == "create_ticket"
    assert action.call_id == "call_7"
    assert action.inputs == {"title": "T", "description": "D"}


def test_two_tool_calls_in_one_answer_come_out_one_by_one():
    brain, client = make_brain(
        [
            tool_answer(
                [
                    ("call_1", "search_help_docs", '{"query": "vpn"}'),
                    ("call_2", "create_ticket", '{"title": "T", "description": "D"}'),
                ]
            )
        ]
    )

    first = brain.next_action(a_conversation(), TOOLS)
    second = brain.next_action(a_conversation(), TOOLS)

    assert [first.tool, second.tool] == ["search_help_docs", "create_ticket"]
    assert [first.call_id, second.call_id] == ["call_1", "call_2"]
    # The second call came from the queue, so the model was asked only once.
    assert client.call_count == 1


def test_broken_json_inputs_are_kept_as_raw_text():
    brain, _ = make_brain([tool_answer([("call_1", "create_ticket", "{not json at all")])])

    action = brain.next_action(a_conversation(), TOOLS)

    assert isinstance(action, ToolCall)
    assert action.inputs == {RAW_INPUT_KEY: "{not json at all"}


def test_the_gateway_refuses_raw_inputs_so_the_mistake_is_in_the_trace(fresh_store):
    from agentqa.fake_ticket_system.tool_gateway import ToolGateway
    from agentqa.trace import Trace

    trace = Trace(test_id="demo", acting_user="alice")
    gateway = ToolGateway(store=fresh_store, acting_user="alice", trace=trace)

    outcome = gateway.call("create_ticket", {RAW_INPUT_KEY: "{not json"}, step=1)

    assert not outcome.ok
    assert "Wrong input for create_ticket" in str(outcome.error)
    assert len(trace.failed_calls("create_ticket")) == 1


# ---------- final replies ----------


def test_text_only_becomes_a_final_reply():
    brain, _ = make_brain([text_answer("I made a ticket for you.")])

    action = brain.next_action(a_conversation(), TOOLS)

    assert isinstance(action, FinalReply)
    assert action.reply == "I made a ticket for you."


def test_think_blocks_are_removed_from_the_reply():
    brain, _ = make_brain([text_answer("<think>Let me plan this out.</think>Your ticket is open.")])

    action = brain.next_action(a_conversation(), TOOLS)

    assert action.reply == "Your ticket is open."


@pytest.mark.parametrize(
    "raw, wanted",
    [
        ("<think>a</think>Hello", "Hello"),
        ("Thinking done</think>Hello", "Hello"),
        ("Hello<think>trailing with no end", "Hello"),
        ("<THINK>upper case</THINK>Hello", "Hello"),
        ("No thinking here", "No thinking here"),
    ],
)
def test_strip_thinking_handles_the_shapes_models_produce(raw, wanted):
    assert strip_thinking(raw) == wanted


# ---------- conversation and tool conversion ----------


def test_the_conversation_keeps_tool_calls_and_tool_call_ids():
    conversation = [
        Message(role="system", content="rules"),
        Message(role="user", content="make a ticket"),
        Message(
            role="assistant",
            content="Using tool create_ticket",
            tool="create_ticket",
            tool_call_id="call_9",
            tool_inputs={"title": "T", "description": "D"},
        ),
        Message(role="tool", content='{"ticket": {}}', tool="create_ticket", tool_call_id="call_9"),
    ]

    messages = conversation_as_openai(conversation)

    assert messages[2]["tool_calls"][0]["id"] == "call_9"
    assert messages[2]["tool_calls"][0]["function"]["name"] == "create_ticket"
    assert '"title": "T"' in messages[2]["tool_calls"][0]["function"]["arguments"]
    assert messages[3] == {"role": "tool", "tool_call_id": "call_9", "content": '{"ticket": {}}'}


def test_the_tools_are_sent_in_the_openai_shape():
    sent = tools_as_openai(TOOLS)

    assert [one["function"]["name"] for one in sent] == ["search_help_docs", "create_ticket"]
    assert sent[0]["type"] == "function"
    assert "query" in sent[0]["function"]["parameters"]["properties"]
    # Keywords some providers refuse are taken out. The gateway still checks input.
    assert "additionalProperties" not in sent[1]["function"]["parameters"]


def test_a_field_called_title_is_kept():
    """create_ticket has a field named title. Dropping it would break the tool."""
    parameters = tools_as_openai(TOOLS)[1]["function"]["parameters"]

    assert "title" in parameters["properties"]
    assert parameters["properties"]["title"]["type"] == "string"


def test_every_required_field_is_also_a_property():
    """A schema that asks for a field it never describes is refused by providers."""
    for sent in tools_as_openai(tool_specs()):
        parameters = sent["function"]["parameters"]
        described = set(parameters.get("properties", {}))
        for needed in parameters.get("required", []):
            assert needed in described, sent["function"]["name"] + " lost " + needed


def test_only_keywords_are_dropped_never_field_names():
    schema = {
        "title": "CreateTicketInput",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"title": "Title", "type": "string"},
            "default": {"title": "Default", "type": "string"},
        },
        "required": ["title", "default"],
    }

    cleaned = clean_schema(schema)

    assert "title" not in cleaned          # the keyword went
    assert "additionalProperties" not in cleaned
    assert set(cleaned["properties"]) == {"title", "default"}  # the names stayed
    assert cleaned["properties"]["title"] == {"type": "string"}


def test_the_request_asks_for_one_tool_call_at_a_time_at_temperature_zero():
    brain, client = make_brain([text_answer("done")])

    brain.next_action(a_conversation(), TOOLS)

    sent = client.requests[0]
    assert sent["temperature"] == 0.0
    assert sent["parallel_tool_calls"] is False
    assert sent["model"] == "test-model"


# ---------- usage ----------


def test_usage_totals_add_up_over_several_calls():
    brain, _ = make_brain(
        [
            tool_answer([("call_1", "search_help_docs", '{"query": "vpn"}')], 100, 20),
            text_answer("All done.", 150, 30),
        ]
    )

    brain.next_action(a_conversation(), TOOLS)
    brain.next_action(a_conversation(), TOOLS)
    usage = brain.usage_totals()

    assert usage.prompt_tokens == 250
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 300
    assert usage.calls == 2


# ---------- failures ----------


def _api_connection_error():
    return openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid/v1"))


def test_a_failed_call_becomes_a_brain_error_with_a_plain_message():
    brain, _ = make_brain([_api_connection_error()])

    with pytest.raises(BrainError) as problem:
        brain.next_action(a_conversation(), TOOLS)

    assert str(problem.value) == "Groq did not answer after 3 tries."


def test_a_rate_limit_becomes_a_brain_error_too():
    answer = httpx.Response(429, request=httpx.Request("POST", "https://example.invalid/v1"))
    brain, _ = make_brain([openai.RateLimitError("slow down", response=answer, body=None)])

    with pytest.raises(BrainError) as problem:
        brain.next_action(a_conversation(), TOOLS)

    assert "rate limit" in str(problem.value)


def test_an_error_message_never_shows_the_key():
    client = FakeOpenAIClient(answers=[ValueError("bad key sk-secret-12345 was refused")])
    brain = OpenAICompatibleBrain(PRESET, model="test-model", client=client)
    brain._secret = "sk-secret-12345"

    with pytest.raises(BrainError) as problem:
        brain.next_action(a_conversation(), TOOLS)

    assert "sk-secret-12345" not in str(problem.value)
    assert "***" in str(problem.value)


def test_a_whole_run_sends_a_well_formed_conversation_back_to_the_model():
    """Every assistant tool call must be answered by a tool message with the same id."""
    from fake_openai_client import tool_answer as make_tool_answer

    from agentqa.project_paths import TEST_CASES_DIR
    from agentqa.run_all_tests import run_one_test
    from agentqa.test_case_loader import load_test_cases

    case = [one for one in load_test_cases(TEST_CASES_DIR) if one.id == "printer-ticket"][0]
    client = FakeOpenAIClient(
        answers=[
            make_tool_answer([("call_a", "search_help_docs", '{"query": "printer"}')]),
            make_tool_answer(
                [("call_b", "create_ticket", '{"title": "Printer", "description": "Not printing"}')]
            ),
            text_answer("I made a ticket for your printer problem."),
        ]
    )
    brain = OpenAICompatibleBrain(PRESET, model="test-model", client=client)

    outcome = run_one_test(case, brain=brain)
    last_messages = client.requests[-1]["messages"]
    call_ids = [
        one["tool_calls"][0]["id"] for one in last_messages if one.get("tool_calls")
    ]
    answer_ids = [one["tool_call_id"] for one in last_messages if one["role"] == "tool"]

    assert outcome.passed, outcome.first_failure_message
    assert call_ids == ["call_a", "call_b"]
    assert answer_ids == ["call_a", "call_b"]
    assert last_messages[0]["role"] == "system"
    assert "role: employee" in last_messages[0]["content"]


def test_a_four_hundred_says_what_the_provider_objected_to():
    """A 400 means AgentQA asked wrongly. Hiding the reason makes a run undebuggable."""
    answer = httpx.Response(400, request=httpx.Request("POST", "https://example.invalid/v1"))
    brain, _ = make_brain(
        [openai.BadRequestError("Invalid JSON payload: unknown field", response=answer, body=None)]
    )
    brain._send_parallel_flag = False
    brain._extra_body = {}

    with pytest.raises(BrainError) as problem:
        brain.next_action(a_conversation(), TOOLS)

    assert "400" in str(problem.value)
    assert "unknown field" in str(problem.value)


def test_an_unsupported_optional_field_is_dropped_and_the_call_is_retried():
    """Some providers refuse parallel_tool_calls or the thinking switch."""
    answer = httpx.Response(400, request=httpx.Request("POST", "https://example.invalid/v1"))
    brain, client = make_brain(
        [
            openai.BadRequestError("parallel_tool_calls is not supported", response=answer, body=None),
            text_answer("I made a ticket."),
        ]
    )

    action = brain.next_action(a_conversation(), TOOLS)

    assert isinstance(action, FinalReply)
    assert client.call_count == 2
    assert "parallel_tool_calls" in client.requests[0]
    assert "parallel_tool_calls" not in client.requests[1]


def test_the_providers_own_tool_call_extra_is_carried_back():
    """Gemini 3 hangs a thought_signature on its tool calls and refuses the next
    request without it. AgentQA never reads it, it only carries it."""
    from fake_openai_client import FakeToolCall

    signature = {"google": {"thought_signature": "abc123"}}
    raw = FakeToolCall(
        id="call_1",
        function=FakeFunction(name="get_ticket", arguments='{"ticket_id": "TCK-1001"}'),
    )
    raw.extra_content = signature

    call = tool_call_from_model(raw)
    message = Message(
        role="assistant",
        content="Using tool get_ticket",
        tool="get_ticket",
        tool_call_id=call.call_id,
        tool_inputs=call.inputs,
        tool_extra=call.provider_extra,
    )
    sent = conversation_as_openai([message])

    assert call.provider_extra == signature
    assert sent[0]["tool_calls"][0]["extra_content"] == signature


def test_a_tool_call_with_no_extra_sends_no_extra():
    """Groq and Ollama attach nothing, and the request must stay clean."""
    brain, _ = make_brain([tool_answer([("call_1", "get_ticket", '{"ticket_id": "TCK-1001"}')])])

    call = brain.next_action(a_conversation(), TOOLS)
    sent = conversation_as_openai(
        [
            Message(
                role="assistant",
                content="x",
                tool=call.tool,
                tool_call_id=call.call_id,
                tool_inputs=call.inputs,
                tool_extra=call.provider_extra,
            )
        ]
    )

    assert call.provider_extra is None
    assert "extra_content" not in sent[0]["tool_calls"][0]


def test_the_reason_is_dug_out_of_a_nested_error_body():
    answer = httpx.Response(400, request=httpx.Request("POST", "https://example.invalid/v1"))
    body = [{"error": {"code": 400, "message": "Function call is missing a thought_signature."}}]
    brain, _ = make_brain([openai.BadRequestError("Error code: 400", response=answer, body=body)])
    brain._send_parallel_flag = False
    brain._extra_body = {}

    with pytest.raises(BrainError) as problem:
        brain.next_action(a_conversation(), TOOLS)

    assert "thought_signature" in str(problem.value)
