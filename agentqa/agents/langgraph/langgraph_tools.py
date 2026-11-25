"""The gateway's tools, as LangChain tools.

This is the whole idea of Phase 5. AgentQA does not control how a LangGraph
agent thinks. It owns the tools. Every tool the agent calls lands in
`gateway.call`, so it is input checked, permission checked and written into the
trace, exactly as a builtin run is.

Two rules are kept from the builtin loop:

* Only the tools the acting user's role may use are built, which is the same
  least privilege a real system has. The gateway still checks every call,
  because an agent can always ask for a tool it was never shown.
* The tool's input schema is the existing pydantic input model, so the model
  sees the same field names, descriptions and allowed values it sees today. It
  is handed over as that model's own JSON schema rather than as the class,
  because a LangChain tool built from the class would validate the inputs
  itself and refuse a bad call before the gateway ever saw it. AgentQA has to
  see that call: the builtin agent sends bad inputs to the gateway, the gateway
  records a tool_error, and the trace shows the mistake. Both agents must land
  in the same place, or a comparison between them is measuring the wrapper.

One step counter is shared by every tool in a run, so the step numbers in the
trace read the way they always have: one step per tool call.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from agentqa.agent_loop import tools_the_user_may_use
from agentqa.brains.openai_compatible_brain import clean_schema
from agentqa.fake_ticket_system.it_support_tools import TOOLS
from agentqa.fake_ticket_system.tool_gateway import ToolGateway

# The tools that only read. A guard does not need to look at these.
READ_ONLY_TOOLS = ("search_help_docs", "get_ticket")

# The tools that change the ticket system. These are what a guard checks.
WRITE_TOOLS = ("create_ticket", "update_ticket", "assign_ticket", "send_notification")


def is_write_tool(tool_name: str) -> bool:
    return tool_name in WRITE_TOOLS


class StepCounter:
    """The step number of the next tool call in this run.

    The builtin loop counts one step per turn. A LangGraph agent has no turn
    number of its own, so the run keeps one counter and every tool call takes
    the next number from it.
    """

    def __init__(self, first: int = 1) -> None:
        self.step = first

    def take(self) -> int:
        number = self.step
        self.step += 1
        return number

    def peek(self) -> int:
        return self.step


def call_the_gateway(
    gateway: ToolGateway,
    counter: StepCounter,
    tool_name: str,
    inputs: dict[str, Any],
) -> str:
    """One gateway call, as the text the agent gets back.

    Nothing else in this package may touch the ticket store. This is the door.
    """
    outcome = gateway.call(tool_name, dict(inputs or {}), counter.take())
    return outcome.as_text()


def _one_tool(gateway: ToolGateway, counter: StepCounter, tool_name: str) -> StructuredTool:
    tool = TOOLS[tool_name]

    def run_it(**inputs: Any) -> str:
        return call_the_gateway(gateway, counter, tool_name, inputs)

    return StructuredTool.from_function(
        func=run_it,
        name=tool.name,
        description=tool.description,
        # The same schema the builtin brain is shown, cleaned the same way, so
        # neither agent has an easier tool to call than the other.
        args_schema=clean_schema(tool.input_model.model_json_schema()),
    )


def build_gateway_tools(
    gateway: ToolGateway,
    role: str,
    counter: StepCounter,
) -> list[StructuredTool]:
    """One LangChain tool per tool this role may use, always in the same order."""
    return [_one_tool(gateway, counter, spec.name) for spec in tools_the_user_may_use(role)]


def tool_names_of(tools: list[StructuredTool]) -> list[str]:
    return [str(one.name) for one in tools]
