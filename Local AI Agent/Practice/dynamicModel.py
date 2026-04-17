import time
from typing import Callable
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langsmith import traceable
from hookRegistry import HookRegistry, ModelRequest, ModelResponse
from modelFactory import ModelFactory
from loadEnv import (
    SYSTEM_PROMPT, CODE_PROMPT, MAX_RETRIES,
)

# One shared factory — reads threshold from .env at startup
factory = ModelFactory()

# ---------------------------------------------------------------------------
# 1. wrap_model_call decorator
# ---------------------------------------------------------------------------

def wrap_model_call(fn: Callable) -> Callable:
    def wrapper(request: ModelRequest, handler: Callable) -> ModelResponse:
        return fn(request, handler)
    return wrapper

# ---------------------------------------------------------------------------
# 2. run_middleware_chain — Chain of Responsibility + retry at the LLM call
# ---------------------------------------------------------------------------

@traceable(name="middleware_chain")
def run_middleware_chain(request: ModelRequest, middleware: list[Callable]) -> ModelResponse:
    def call_next(i: int, req: ModelRequest) -> ModelResponse:
        if i == len(middleware):
            for attempt in range(MAX_RETRIES):
                try:
                    return req.model.invoke(req.messages)
                except Exception:
                    if attempt < MAX_RETRIES - 1:
                        pass
                    else:
                        raise
        return middleware[i](req, lambda r: call_next(i + 1, r))

    return call_next(0, request)

# ---------------------------------------------------------------------------
# 3. dynamic_model_selection — middleware
#    Uses factory.create() instead of referencing global model instances.
# ---------------------------------------------------------------------------

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler: Callable) -> ModelResponse:
    from langchain_core.runnables.base import RunnableBinding
    # If agent_node already bound tools, the model is a RunnableBinding —
    # don't override it or the tool binding (and model upgrade) will be lost.
    if not isinstance(request.model, RunnableBinding):
        human_turns = sum(isinstance(m, HumanMessage) for m in request.state["messages"])
        request.model = factory.create(human_turns)
    return handler(request)

# ---------------------------------------------------------------------------
# 4. create_agent
#
# Template Method — _run_node() holds the shared skeleton for every node:
#   emit before_agent → build request → run chain → emit after_model/on_error → emit after_agent
#
# agent_node and code_agent_node are now 2 lines each — they only supply
# the parts that differ: which model and which system prompt.
# ---------------------------------------------------------------------------


def create_agent(middleware: list[Callable], hooks: HookRegistry, tools: list | None = None):
    _tools = tools or []

    # ----- Template Method: shared node skeleton -----
    def _run_node(state: MessagesState, model, system_prompt: str, node_middleware: list) -> dict:
        hooks.emit("before_agent", state)
        request = ModelRequest(
            model=model.bind_tools(_tools),
            messages=[SystemMessage(content=system_prompt)] + state["messages"],
            state=state,
        )
        try:
            hooks.emit("before_model", request)
            response = (
                run_middleware_chain(request, node_middleware)
                if node_middleware
                else request.model.invoke(request.messages)
            )
            hooks.emit("after_model", response)
        except Exception as exc:
            hooks.emit("on_error", exc, request)
            raise
        hooks.emit("after_agent", state)
        return {"messages": [response]}

    # ----- Nodes: only the variable parts -----
    # llama3.2:3b too bad to handle too call
    def agent_node(state: MessagesState) -> dict:
        human_turns = sum(isinstance(m, HumanMessage) for m in state["messages"])
        return _run_node(state, factory.create(human_turns, has_tools=bool(_tools)), SYSTEM_PROMPT, middleware)

    def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    checkpointer = MemorySaver()
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(_tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)

# ---------------------------------------------------------------------------
# 5. Demo — uses AgentBuilder to show the pattern end-to-end
# ---------------------------------------------------------------------------

def main():
    from agentBuilder import AgentBuilder

    agent = (AgentBuilder()
        .with_middleware(dynamic_model_selection)
        .build())

    config = {"configurable": {"thread_id": "demo-thread"}}
    conversation = [
        "What is machine learning?",
        "Can you give me an example?",
        "How is that different from deep learning?",
        "What is the weather like in Edmonton, Canada?",
        "And what is attention mechanism?",
    ]

    print("Starting multi-turn conversation...\n")
    for turn, user_input in enumerate(conversation, start=1):
        print(f"\n--- Turn {turn} ---")
        print(f"User: {user_input}")
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        reply = result["messages"][-1].content
        print(f"Agent: {reply[:120]}{'...' if len(reply) > 120 else ''}")

if __name__ == "__main__":
    main()
