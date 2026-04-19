# Local AI Agent — LangChain Middleware Examples
From https://www.youtube.com/watch?v=J7j5tCB_y4w

## Message Types

LangChain uses three message classes that map to the roles in a chat conversation.

| Class | Role | Represents |
|---|---|---|
| `SystemMessage` | `"system"` | Developer instructions — rules set before the conversation starts |
| `HumanMessage` | `"user"` | What the user typed |
| `AIMessage` | `"assistant"` | What the model replied |

## System Prompt Template

References:
- [Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide)
- [Claude Prompting Best Practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)

## LangGraph Flow

```
YOUR CODE                    LANGCHAIN                    LANGGRAPH
─────────────────────────────────────────────────────────────────

agent.ainvoke()  ──────────────────────────────►  StateGraph
                                                      │
                                                   START
                                                      │ add_edge
                                                      ▼
agent_node()  ◄── add_node ──────────────────── "agent"
    │                                               │
    └── LLM.ainvoke()  ──── ChatOllama ─────────►  │
        bind_tools()   ──── RunnableBinding ─────►  │
                                                      │ add_conditional_edges
                                                      ▼
should_continue() ◄── you write this ────────── router
    │
    ├── "tools" ──────────────────────────────── "tools"  ◄── add_node
    │               ToolNode.ainvoke() ◄── ToolNode (prebuilt)
    │               add_edge("tools","agent") ──► loops back
    │
    └── END ──────────────────────────────────── graph stops


KEY LANGCHAIN:  Runnable / RunnableBinding / ChatOllama / ToolNode
KEY LANGGRAPH:  StateGraph / MessagesState / START / END / MemorySaver
YOU WRITE:      agent_node / logged_tool_node / should_continue
```

## Async Gotcha — nested `asyncio.run()`

**Symptom:** duplicate output or recursive traceback from `asyncio\runners.py`.

**Cause:** `asyncio.run()` called inside an `async def` that is already running under `asyncio.run()`.

```
YOUR CODE (WRONG)                          YOUR CODE (CORRECT)
────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":                 if __name__ == "__main__":
    asyncio.run(run_for_user("123"))           asyncio.run(run_for_user("123"))
         │                                              │
         │  starts event loop                          │  starts ONE event loop
         ▼                                              ▼
async def run_for_user():                  async def run_for_user():
    ...                                        ...
    result = asyncio.run(                      result = await agent.ainvoke(
        agent.ainvoke(...)                         {"messages": [...]},
    )                                          )
    ↑ WRONG: starts a second                   ↑ CORRECT: already inside loop,
      event loop inside the first                just await the coroutine
      → duplicate output / recursion
      crash in asyncio\runners.py
```

**Rule:** one `asyncio.run()` at `if __name__ == "__main__"`. Use `await` everywhere else inside async functions.
