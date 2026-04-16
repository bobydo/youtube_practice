from dataclasses import dataclass
from typing import Callable
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

# Type alias — mirrors the tutorial's ModelResponse
ModelResponse = BaseMessage

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASIC_MODEL    = "llama3.2:3b"
ADVANCED_MODEL = "qwen3:8b"
SYSTEM_PROMPT  = "You are a helpful assistant. Answer clearly and concisely."

# Models instantiated once — not recreated on every call
basic_model    = ChatOllama(model=BASIC_MODEL)
advanced_model = ChatOllama(model=ADVANCED_MODEL)

# ---------------------------------------------------------------------------
# 1. ModelRequest — carrier passed through the middleware chain
# ---------------------------------------------------------------------------

@dataclass
class ModelRequest:
    model: ChatOllama     # middleware can swap this before handler runs
    messages: list        # full message list sent to the LLM
    state: MessagesState  # LangGraph state — lets middleware read history length

# ---------------------------------------------------------------------------
# 2. wrap_model_call decorator
#    Wraps a function (request, handler) -> ModelResponse.
#    handler does the actual LLM call with whatever model is set on request.
# ---------------------------------------------------------------------------

def wrap_model_call(fn: Callable) -> Callable:
    def handler(request: ModelRequest) -> ModelResponse:
        return request.model.invoke(request.messages)

    def wrapper(request: ModelRequest) -> ModelResponse:
        return fn(request, handler)

    return wrapper

# ---------------------------------------------------------------------------
# 3. dynamic_model_selection — middleware logic
#    Reads message count from LangGraph state, picks model, delegates to handler.
# ---------------------------------------------------------------------------

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state["messages"])

    if message_count > 3:
        model = advanced_model
    else:
        model = basic_model

    request.model = model
    print(f"  [wrap_model_call] {message_count} messages → {model.model}")
    return handler(request)

# ---------------------------------------------------------------------------
# 4. create_agent — StateGraph compiled with MemorySaver checkpointer
#    agent_node is a closure so it has access to the middleware list
# ---------------------------------------------------------------------------

def create_agent(middleware: list[Callable]):
    def agent_node(state: MessagesState) -> dict:
        request = ModelRequest(
            model=basic_model,
            messages=[SystemMessage(content=SYSTEM_PROMPT)] + state["messages"],
            state=state,
        )
        # run through each middleware in order; each calls handler internally
        response = middleware[0](request) if middleware else basic_model.invoke(request.messages)
        return {"messages": [response]}

    checkpointer = MemorySaver()
    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=checkpointer)

# ---------------------------------------------------------------------------
# 6. Demo — multi-turn conversation, history managed by MemorySaver
# ---------------------------------------------------------------------------

def main():
    agent = create_agent(middleware=[dynamic_model_selection])
    config = {"configurable": {"thread_id": "demo-thread"}}

    conversation = [
        "What is machine learning?",
        "Can you give me an example?",
        "How is that different from deep learning?",
        "What are transformers in that context?",   # crosses threshold here
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
