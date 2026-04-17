"""
agentBuilder.py — Builder Pattern for constructing the agent graph.

Replaces the create_agent() function call with a fluent interface.
Each .with_*() method configures one concern and returns self for chaining.
.build() wires everything together and returns the compiled LangGraph.

Usage:
    agent = (AgentBuilder()
        .with_middleware(dynamic_model_selection)
        .with_tools(locate_user, get_weather)
        .build())
"""
from typing import Callable
from hookRegistry import HookRegistry
from dynamicModel import create_agent, dynamic_model_selection


class AgentBuilder:
    def __init__(self):
        self._middleware: list[Callable] = []
        self._tools: list = []
        self._hooks: HookRegistry = HookRegistry()   # default hooks registered automatically

    def with_middleware(self, *fns: Callable) -> "AgentBuilder":
        """Add one or more middleware functions to the chain."""
        self._middleware.extend(fns)
        return self

    def with_tools(self, *tools) -> "AgentBuilder":
        """Add one or more LangChain tools the LLM can call."""
        self._tools.extend(tools)
        return self

    def with_hooks(self, hooks: HookRegistry) -> "AgentBuilder":
        """Replace the default HookRegistry with a custom one."""
        self._hooks = hooks
        return self

    def build(self):
        """Compile and return the LangGraph agent."""
        return create_agent(
            middleware=self._middleware,
            hooks=self._hooks,
            tools=self._tools,
        )
