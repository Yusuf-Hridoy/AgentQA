"""Picking an agent, and failing early when one cannot be used.

A setup problem has to stop the run before the first test. Ten tests marked
ERROR for one missing package is noise, not a result.
"""

from __future__ import annotations

import sys

import pytest
from typer.testing import CliRunner

from agentqa.agents.agent_factory import (
    MISSING_EXTRA,
    UnknownAgentError,
    agent_framework_label,
    build_agent,
    check_agent_ready,
    guard_prompt_version_for,
    import_langgraph,
)
from agentqa.agents.agent_interface import (
    AGENT_BUILTIN,
    AGENT_LANGGRAPH_GUARD,
    AGENT_LANGGRAPH_REACT,
    NO_FRAMEWORK,
    AgentSetupError,
)
from agentqa.agents.builtin_agent import BuiltinAgent
from agentqa.cli import SETUP_ERROR_EXIT_CODE, app

langgraph = pytest.importorskip("langgraph")

# The modules a LangGraph agent needs. The missing-extra test hides them.
LANGGRAPH_MODULES = (
    "agentqa.agents.langgraph.langgraph_model",
    "agentqa.agents.langgraph.langgraph_react_agent",
    "agentqa.agents.langgraph.langgraph_guard_agent",
)


def test_the_default_agent_is_the_builtin_one():
    built = build_agent(test_id="printer-ticket")
    assert isinstance(built, BuiltinAgent)
    assert built.name == AGENT_BUILTIN


def test_an_unknown_agent_name_is_a_setup_error():
    with pytest.raises(UnknownAgentError) as problem:
        build_agent("langchain-react", test_id="printer-ticket")
    assert "langchain-react" in str(problem.value)
    assert AGENT_LANGGRAPH_REACT in str(problem.value)


def test_the_builtin_agent_says_it_uses_no_framework():
    assert agent_framework_label(AGENT_BUILTIN) == NO_FRAMEWORK
    assert guard_prompt_version_for(AGENT_BUILTIN) is None
    assert guard_prompt_version_for(AGENT_LANGGRAPH_REACT) is None


def test_a_langgraph_agent_records_the_framework_version():
    label = agent_framework_label(AGENT_LANGGRAPH_REACT)
    assert label.startswith("langgraph ")
    assert label != "langgraph "


def test_only_the_guard_agent_has_a_guard_prompt_version():
    assert guard_prompt_version_for(AGENT_LANGGRAPH_GUARD) == "guard-v1"


def test_gemini_cannot_drive_a_langgraph_agent():
    """The thought-signature problem is known, so say so instead of failing ten times."""
    with pytest.raises(AgentSetupError) as problem:
        check_agent_ready(AGENT_LANGGRAPH_REACT, "gemini")
    said = str(problem.value)
    assert "Gemini" in said
    assert "groq or local" in said


def test_gemini_with_a_langgraph_agent_exits_with_the_setup_code(tmp_path):
    result = CliRunner().invoke(
        app,
        ["run", "--agent", AGENT_LANGGRAPH_GUARD, "--brain", "gemini", "--reports-dir", str(tmp_path)],
    )
    assert result.exit_code == SETUP_ERROR_EXIT_CODE
    assert "not supported yet" in result.output


def hide_langgraph(monkeypatch):
    """Make the LangGraph import fail, the way a missing extra makes it fail.

    A module already imported is both an entry in sys.modules and an attribute
    of its package, and `from package import module` reads the attribute first.
    So both have to go, or the import quietly succeeds and the test proves
    nothing.
    """
    import agentqa.agents.langgraph as package

    for name in LANGGRAPH_MODULES:
        monkeypatch.setitem(sys.modules, name, None)
        short = name.rsplit(".", 1)[-1]
        if hasattr(package, short):
            monkeypatch.delattr(package, short)


def test_a_missing_extra_gives_a_clear_message(monkeypatch):
    hide_langgraph(monkeypatch)
    with pytest.raises(AgentSetupError) as problem:
        import_langgraph()
    assert str(problem.value) == MISSING_EXTRA
    assert 'pip install -e ".[langgraph]"' in str(problem.value)


def test_a_missing_extra_is_caught_before_the_first_test(monkeypatch):
    hide_langgraph(monkeypatch)
    with pytest.raises(AgentSetupError):
        check_agent_ready(AGENT_LANGGRAPH_REACT, "fake")


def test_without_the_extra_the_framework_label_falls_back(monkeypatch):
    hide_langgraph(monkeypatch)
    assert agent_framework_label(AGENT_LANGGRAPH_REACT) == NO_FRAMEWORK


def test_a_missing_extra_exits_with_the_setup_code(monkeypatch, tmp_path):
    hide_langgraph(monkeypatch)
    result = CliRunner().invoke(
        app,
        ["run", "--agent", AGENT_LANGGRAPH_REACT, "--brain", "fake", "--reports-dir", str(tmp_path)],
    )
    assert result.exit_code == SETUP_ERROR_EXIT_CODE
    assert "pip install" in result.output
