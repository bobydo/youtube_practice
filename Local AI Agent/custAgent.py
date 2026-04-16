import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

ModelResponse = BaseMessage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL         = "llama3.2:3b"
SYSTEM_PROMPT = "You are a helpful assistant. Answer clearly and concisely."

llm = ChatOllama(model=MODEL)

# ---------------------------------------------------------------------------
# 1. ModelRequest — carrier passed into hook callbacks
# ---------------------------------------------------------------------------

@dataclass
class ModelRequest:
    model: ChatOllama
    messages: list
    state: MessagesState

# ---------------------------------------------------------------------------
# 2. HookRegistry — Observer Pattern
#    Maps event names to lists of registered callbacks.
#    .on()   registers a callback for an event
#    .emit() fires all callbacks registered for that event
# ---------------------------------------------------------------------------

class HookRegistry:
    def __init__(self):
        self._hooks: dict[str, list[Callable]] = defaultdict(list)

    def on(self, event: str, callback: Callable) -> None:
        """Register a callback for a named event."""
        self._hooks[event].append(callback)

    def emit(self, event: str, *args) -> None:
        """Fire all callbacks registered for the event, in registration order."""
        for callback in self._hooks[event]:
            callback(*args)

# ---------------------------------------------------------------------------
# 3. Timing callbacks — plain functions registered to hooks
#    State shared via a simple dict (no class needed)
# ---------------------------------------------------------------------------

timing = {}   # shared state across callbacks

def on_before_agent(state: MessagesState) -> None:
    timing["agent_start"] = time.time()
    print(f"  [before_agent] turn started  |  history: {len(state['messages'])} messages")

def on_before_model(request: ModelRequest) -> None:
    timing["model_start"] = time.time()
    print(f"  [before_model] calling {request.model.model}")

def on_after_model(_) -> None:
    elapsed = time.time() - timing["model_start"]
    print(f"  [after_model]  model responded in {elapsed:.2f}s")

def on_after_agent(_) -> None:
    elapsed = time.time() - timing["agent_start"]
    print(f"  [after_agent]  total turn time: {elapsed:.2f}s")

# ---------------------------------------------------------------------------
# 4. create_agent — emits hook events at each lifecycle point
# ---------------------------------------------------------------------------

def create_agent(hooks: HookRegistry):
    def agent_node(state: MessagesState) -> dict:
        hooks.emit("before_agent", state)

        request = ModelRequest(
            model=llm,
            messages=[SystemMessage(content=SYSTEM_PROMPT)] + state["messages"],
            state=state,
        )

        hooks.emit("before_model", request)
        response = request.model.invoke(request.messages)
        hooks.emit("after_model", response)

        hooks.emit("after_agent", state)
        return {"messages": [response]}

    checkpointer = MemorySaver()
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=checkpointer)

# ---------------------------------------------------------------------------
# 5. Demo — register callbacks, then run
# ---------------------------------------------------------------------------

def main():
    hooks = HookRegistry()
    hooks.on("before_agent", on_before_agent)
    hooks.on("before_model", on_before_model)
    hooks.on("after_model",  on_after_model)
    hooks.on("after_agent",  on_after_agent)

    agent = create_agent(hooks=hooks)
    config = {"configurable": {"thread_id": "demo-thread"}}

    conversation = [
        "What is machine learning?",
        "Can you give me an example?",
        "How is that different from deep learning?",
        "What are transformers in that context?",
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
