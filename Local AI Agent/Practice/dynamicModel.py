from email.mime import base
import time
from typing import Callable
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from hookRegistry import HookRegistry, ModelRequest
from modelFactory import ModelFactory
from modelMiddleware import ModelMiddleware
from loadEnv import SYSTEM_PROMPT
from loggerSetup import get_logger

logger = get_logger(__name__)

# One shared factory and middleware instance
factory = ModelFactory()
middleware_instance = ModelMiddleware(factory)

# ---------------------------------------------------------------------------
# 1. create_agent
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
    async def _run_node(state: MessagesState, model, system_prompt: str, node_middleware: list) -> dict:
        hooks.emit("before_agent", state)
        #Runnable  (base interface — ainvoke, invoke, stream)
        #└── RunnableBinding  (wraps a Runnable + bound config/tools)
        #    └── model.bind_tools(...)  returns this
        request = ModelRequest(
            model=model.bind_tools(_tools),
            messages=[SystemMessage(content=system_prompt)] + state["messages"],
            state=state,
        )
        try:
            hooks.emit("before_model", request)
            response = (
                await middleware_instance.run_middleware_chain(request, node_middleware)
                if node_middleware
                else await request.model.ainvoke(request.messages)
            )
            hooks.emit("after_model", response)
        except Exception as exc:
            hooks.emit("on_error", exc, request)
            raise
        hooks.emit("after_agent", state)
        return {"messages": [response]}

    # ----- Nodes: only the variable parts -----
    async def agent_node(state: MessagesState) -> dict:
        human_turns = sum(isinstance(m, HumanMessage) for m in state["messages"])
        return await _run_node(state, factory.create(human_turns), SYSTEM_PROMPT, middleware)

    def should_continue(state: MessagesState) -> str:
        for msg in state["messages"]:
            logger.info("last_message  type=%s content=%s", type(msg).__name__, str(msg.content)[:120])
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    _tool_node = ToolNode(_tools)

    async def logged_tool_node(state: MessagesState) -> dict:
        last = state["messages"][-1]
        for tc in getattr(last, "tool_calls", []):
            logger.info("tool_call    name=%s  args=%s", tc["name"], tc["args"])
        # get_weather + get_news will run parallel if both called
        result = await _tool_node.ainvoke(state) 
        for msg in result.get("messages", []):
            logger.info("logged_tool_node => tool_result  name=%s  content=%s",
                        getattr(msg, "name", "?"), str(msg.content)[:120])
        return result

    # MessagesState uses add_messages which appends, never replaces:
    # # Turn 1 messages: [HumanMessage("what is ML?"), AIMessage(tool_calls=[locate_user])]
    # After logged_tool_node:
    # messages: [HumanMessage, AIMessage(tool_calls), ToolMessage("Edmonton")]
    # # After agent_node again:
    # messages: [HumanMessage, AIMessage(tool_calls), ToolMessage, AIMessage("The weather is...")]
    checkpointer = MemorySaver()
    
    graph = StateGraph(MessagesState)
    graph.add_edge(START, "agent")
    graph.add_node("agent", agent_node)
    graph.add_node("tools", logged_tool_node)
    graph.add_conditional_edges("agent", should_continue)
    # After logged_tool_node finishes, always go back to agent_node.
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)

# ---------------------------------------------------------------------------
# 5. Demo — uses AgentBuilder to show the pattern end-to-end
# ---------------------------------------------------------------------------

def main():
    from agentBuilder import AgentBuilder

    agent = (AgentBuilder()
        .with_middleware(middleware_instance.dynamic_model_selection)
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
        print(f"Agent: {reply}")

if __name__ == "__main__":
    main()
