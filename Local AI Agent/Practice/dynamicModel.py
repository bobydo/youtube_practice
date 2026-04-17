import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langsmith import traceable
from hookRegistry import HookRegistry, ModelRequest
from loadEnv import (
    BASIC_MODEL, ADVANCED_MODEL, CODE_MODEL,
    SYSTEM_PROMPT, CODE_PROMPT,
    COMPLEXITY_THRESHOLD, MAX_RETRIES,
)

ModelResponse = BaseMessage

# ---------------------------------------------------------------------------
# Model factory — creates a fresh instance per request based on turn count.
# No global model singletons: the right model is chosen at call time.
# ---------------------------------------------------------------------------

def get_model(turns: int) -> ChatOllama:
    """Return a fresh model instance based on conversation turn count."""
    if turns > COMPLEXITY_THRESHOLD:
        return ChatOllama(model=ADVANCED_MODEL)
    return ChatOllama(model=BASIC_MODEL)


# ---------------------------------------------------------------------------
# 4. wrap_model_call decorator
# ---------------------------------------------------------------------------

def wrap_model_call(fn: Callable) -> Callable:
    def wrapper(request: ModelRequest, handler: Callable) -> ModelResponse:
        return fn(request, handler)
    return wrapper

# ---------------------------------------------------------------------------
# 5. run_middleware_chain — retry logic at the LLM call
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
# 6. dynamic_model_selection — middleware
#
# Uses get_model() instead of global instances.
# Note: agent_node also calls get_model() + bind_tools before entering the
# chain, so this middleware's model swap is overridden by the bound model.
# It is kept here to show the middleware pattern.
# ---------------------------------------------------------------------------

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler: Callable) -> ModelResponse:
    human_turns = sum(isinstance(m, HumanMessage) for m in request.state["messages"])
    request.model = get_model(human_turns)
    return handler(request)

# ---------------------------------------------------------------------------
# 7. create_agent
#
# agent_node flow:
#   1. compute human turn count
#   2. get_model(turns) — picks basic or advanced
#   3. .bind_tools(tools) — caller passes tools; LLM decides when to use them
#   4. run through middleware chain (handles retries + hooks)
# ---------------------------------------------------------------------------

def route_by_content(state: MessagesState) -> str:
    last_message = str(state["messages"][-1].content).lower()
    return "code_agent" if "code" in last_message else "agent"

def create_agent(middleware: list[Callable], hooks: HookRegistry, tools: list | None = None):
    _tools = tools or []   # caller passes tools; library stays tool-agnostic

    def agent_node(state: MessagesState) -> dict:
        hooks.emit("before_agent", state)
        human_turns = sum(isinstance(m, HumanMessage) for m in state["messages"])
        # bind_tools gives the LLM awareness of whatever tools the caller provided
        model = get_model(human_turns).bind_tools(_tools)
        request = ModelRequest(
            model=model,
            messages=[SystemMessage(content=SYSTEM_PROMPT)] + state["messages"],
            state=state,
        )
        try:
            hooks.emit("before_model", request)
            response = run_middleware_chain(request, middleware) if middleware else model.invoke(request.messages)
            hooks.emit("after_model", response)
        except Exception as exc:
            hooks.emit("on_error", exc, request)
            raise
        hooks.emit("after_agent", state)
        return {"messages": [response]}

    def code_agent_node(state: MessagesState) -> dict:
        hooks.emit("before_agent", state)
        model = ChatOllama(model=CODE_MODEL).bind_tools(_tools)
        request = ModelRequest(
            model=model,
            messages=[SystemMessage(content=CODE_PROMPT)] + state["messages"],
            state=state,
        )
        try:
            hooks.emit("before_model", request)
            response = run_middleware_chain(request, [])
            hooks.emit("after_model", response)
        except Exception as exc:
            hooks.emit("on_error", exc, request)
            raise
        hooks.emit("after_agent", state)
        return {"messages": [response]}

    checkpointer = MemorySaver()
    graph = StateGraph(MessagesState)
    graph.add_node("agent",      agent_node)
    graph.add_node("code_agent", code_agent_node)
    graph.add_conditional_edges(START, route_by_content, {
        "agent":      "agent",
        "code_agent": "code_agent",
    })
    graph.add_edge("agent",      END)
    graph.add_edge("code_agent", END)
    return graph.compile(checkpointer=checkpointer)

# ---------------------------------------------------------------------------
# 8. Demo
# ---------------------------------------------------------------------------

def main():
    hooks = HookRegistry()

    agent = create_agent(middleware=[dynamic_model_selection], hooks=hooks, tools=[])
    config = {"configurable": {"thread_id": "demo-thread"}}

    conversation = [
        "What is machine learning?",
        "Can you give me an example?",
        "How is that different from deep learning?",
        "What is the weather like in Edmonton, Canada?",   # LLM calls get_weather automatically
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
