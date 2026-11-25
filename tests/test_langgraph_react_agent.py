"""The plain LangGraph ReAct agent, driven by the scripted fake model.

No key, no network. The point of these tests is that an agent someone else
built lands in the same trace, the same checks and the same report as
AgentQA's own loop.
"""

from __future__ import annotations

import pytest

from agentqa.agents.agent_interface import AGENT_LANGGRAPH_REACT
from agentqa.run_all_tests import run_one_test
from agentqa.test_case_loader import load_test_cases
from agentqa.trace import FINAL_REPLY, STEP_LIMIT_HIT

pytest.importorskip("langgraph")

from agentqa.agents.langgraph.langgraph_fake_model import FakeChatModel  # noqa: E402
from agentqa.agents.langgraph.langgraph_react_agent import (  # noqa: E402
    LangGraphReactAgent,
    recursion_limit_for,
    system_prompt_for,
)


def one_case(test_id: str):
    return load_test_cases(None, only=test_id)[0]


def test_every_test_passes_with_the_fake_model():
    """10 of 10, the same as the builtin agent on the same scripts."""
    passed = [
        case.id
        for case in load_test_cases(None)
        if run_one_test(case, brain_name="fake", agent_name=AGENT_LANGGRAPH_REACT).passed
    ]
    assert len(passed) == 10


def test_it_lands_in_the_same_trace_the_builtin_agent_writes():
    case = one_case("printer-ticket")
    builtin = run_one_test(case, brain_name="fake")
    react = run_one_test(case, brain_name="fake", agent_name=AGENT_LANGGRAPH_REACT)

    assert react.trace.called_tool_names_in_order() == builtin.trace.called_tool_names_in_order()
    assert react.ticket_system["new_ticket_ids"] == builtin.ticket_system["new_ticket_ids"]
    assert react.final_reply == builtin.final_reply


def test_the_system_message_is_the_builtin_one_word_for_word():
    from agentqa.agent_loop import AGENT_RULES

    case = one_case("printer-ticket")
    said = system_prompt_for(case, "employee")
    assert said.startswith(AGENT_RULES)
    assert said.endswith("You are helping alice (role: employee).")


def test_the_step_limit_leaves_room_for_the_turn_that_answers():
    """Two nodes per step, and one more model node to answer with."""
    assert recursion_limit_for(8) == 17
    assert recursion_limit_for(1) == 3


def test_running_out_of_steps_is_recorded_and_nothing_is_invented(tmp_path):
    """An agent that never answers has no reply. We do not write one for it."""
    script = tmp_path / "loop.yaml"
    script.write_text(
        "actions:\n"
        + "".join(
            "  - tool: search_help_docs\n    inputs: { query: \"again\" }\n"
            for _ in range(40)
        ),
        encoding="utf-8",
    )
    case = one_case("printer-ticket").model_copy(update={"max_steps": 3})
    agent = LangGraphReactAgent(FakeChatModel.from_script_file(script))

    outcome = run_one_test(case, agent=agent)

    assert outcome.trace.hit_step_limit()
    assert outcome.trace.events_of_kind(STEP_LIMIT_HIT)
    assert outcome.final_reply is None
    assert not outcome.passed


def test_an_empty_reply_is_recorded_as_empty(tmp_path):
    """A model that answers with nothing is reported honestly, not padded."""
    from langchain_core.messages import AIMessage

    case = one_case("printer-ticket")

    class EmptyModel(FakeChatModel):
        """Answers with no tool call and no text at all."""

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.outputs import ChatGeneration, ChatResult

            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

    outcome = run_one_test(case, agent=LangGraphReactAgent(EmptyModel()))

    replies = outcome.trace.events_of_kind(FINAL_REPLY)
    assert replies[-1].empty_reply is True
    assert outcome.final_reply == ""
    assert not outcome.passed


def test_a_langgraph_run_has_no_guard_to_report():
    case = one_case("printer-ticket")
    outcome = run_one_test(case, brain_name="fake", agent_name=AGENT_LANGGRAPH_REACT)
    assert outcome.guard_blocks == []
    assert outcome.guard_stopped == []
    assert outcome.guard_wrongly_blocked == []
    assert outcome.guard_tokens.total_tokens == 0
