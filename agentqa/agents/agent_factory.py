"""Picks the agent for a run, and says early when one cannot be used.

Everything that can fail before the first test fails before the first test. A
missing extra, or a brain a LangGraph agent cannot drive, is a setup error with
exit code 2. Ten tests marked ERROR for the same reason would be noise.

LangGraph is imported here and nowhere above here, so the core AgentQA installs
and runs with no LangGraph on the machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentqa.agents.agent_interface import (
    AGENT_BUILTIN,
    AGENT_LANGGRAPH_GUARD,
    AGENT_LANGGRAPH_REACT,
    ALL_AGENT_NAMES,
    NO_FRAMEWORK,
    AgentSetupError,
    AgentUnderTest,
    is_langgraph_agent,
)
from agentqa.agents.builtin_agent import BuiltinAgent
from agentqa.brains.brain_factory import BRAIN_FAKE, build_brain
from agentqa.brains.brain_interface import Brain

MISSING_EXTRA = 'LangGraph is not installed. Run: pip install -e ".[langgraph]"'


class UnknownAgentError(AgentSetupError):
    """The agent name is not one AgentQA knows."""


def check_agent_name(agent_name: str) -> None:
    if agent_name not in ALL_AGENT_NAMES:
        raise UnknownAgentError(
            "There is no agent called " + str(agent_name)
            + ". Known agents: " + ", ".join(ALL_AGENT_NAMES) + "."
        )


def import_langgraph() -> Any:
    """The LangGraph pieces, or a clear message about the missing extra."""
    try:
        from agentqa.agents.langgraph import langgraph_guard_agent, langgraph_model, langgraph_react_agent
    except ImportError as problem:
        raise AgentSetupError(MISSING_EXTRA) from problem
    return langgraph_model, langgraph_react_agent, langgraph_guard_agent


def langgraph_version() -> str:
    """The framework version, for run_info. Never guessed."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return "langgraph " + version("langgraph")
    except PackageNotFoundError:
        return "langgraph"


def agent_framework_label(agent_name: str) -> str:
    """What built this agent, as results.json records it."""
    if not is_langgraph_agent(agent_name):
        return NO_FRAMEWORK
    try:
        import_langgraph()
    except AgentSetupError:
        return NO_FRAMEWORK
    return langgraph_version()


def guard_prompt_version_for(agent_name: str) -> str | None:
    """Only the guard agent has a guard prompt."""
    if agent_name != AGENT_LANGGRAPH_GUARD:
        return None
    from agentqa.agents.langgraph.guard_prompt import GUARD_PROMPT_VERSION

    return GUARD_PROMPT_VERSION


def check_agent_ready(agent_name: str, brain_name: str) -> None:
    """Raise AgentSetupError when this agent and brain cannot run a test."""
    check_agent_name(agent_name)
    if not is_langgraph_agent(agent_name):
        return

    langgraph_model, _react, _guard = import_langgraph()
    langgraph_model.check_brain_supported(brain_name)


def _langgraph_model_for(brain_name: str, test_id: str, scripts_dir: str | Path | None) -> Any:
    """The chat model a LangGraph agent thinks with, for one test."""
    langgraph_model, _react, _guard = import_langgraph()
    if brain_name == BRAIN_FAKE:
        from agentqa.agents.langgraph.langgraph_fake_model import FakeChatModel

        return FakeChatModel.for_test_case(test_id, folder=scripts_dir)
    return langgraph_model.build_chat_model(brain_name)


def _guard_for(brain_name: str, model: Any) -> Any:
    """The guard. The fake brain gets the word guard, so CI needs no key."""
    _model, _react, langgraph_guard_agent = import_langgraph()
    if brain_name == BRAIN_FAKE:
        from agentqa.agents.langgraph.fake_guard import FakeGuard

        return FakeGuard()
    return langgraph_guard_agent.ModelGuard(model)


def build_agent(
    agent_name: str = AGENT_BUILTIN,
    brain_name: str = BRAIN_FAKE,
    test_id: str = "",
    brain: Brain | None = None,
    scripts_dir: str | Path | None = None,
) -> AgentUnderTest:
    """One agent for one test. A fresh one each time, as a brain is fresh."""
    check_agent_name(agent_name)

    if agent_name == AGENT_BUILTIN:
        thinking = brain if brain is not None else build_brain(brain_name, test_id, scripts_dir)
        return BuiltinAgent(thinking)

    _model_module, langgraph_react_agent, langgraph_guard_agent = import_langgraph()
    model = _langgraph_model_for(brain_name, test_id, scripts_dir)

    if agent_name == AGENT_LANGGRAPH_REACT:
        return langgraph_react_agent.LangGraphReactAgent(model)

    return langgraph_guard_agent.LangGraphGuardAgent(model, _guard_for(brain_name, model))
