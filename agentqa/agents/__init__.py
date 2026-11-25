"""The agents AgentQA can test: its own loop, or one built with a framework.

AgentQA does not need to control how an agent thinks. It only needs to own the
tools. Every tool an agent calls is a gateway call, so it is permission
checked and written into the trace, whoever built the agent.
"""
