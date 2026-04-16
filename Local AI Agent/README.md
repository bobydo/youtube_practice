# Local AI Agent — LangChain Middleware Examples
From https://www.youtube.com/watch?v=J7j5tCB_y4w

## Message Types

LangChain uses three message classes that map to the roles in a chat conversation.

| Class | Role | Represents |
|---|---|---|
| `SystemMessage` | `"system"` | Developer instructions — rules set before the conversation starts |
| `HumanMessage` | `"user"` | What the user typed |
| `AIMessage` | `"assistant"` | What the model replied |

### When to use each

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),   # always first, set once

    HumanMessage(content="What is PCA?"),                    # user turn 1
    AIMessage(content="PCA reduces dimensions by finding principal components."),  # AI turn 1

    HumanMessage(content="Can you give an example?"),        # user turn 2
    AIMessage(content="Imagine you have 100 features..."),   # AI turn 2

    HumanMessage(content="How is it different from t-SNE?"), # current user message
]
```

The model receives the **entire list** on every call — this is how it remembers the conversation.

### Why the wrong type breaks things

The model expects a strict pattern: `System → Human → AI → Human → AI`.
Using `SystemMessage` to store an AI reply breaks that pattern:

```python
# WRONG — confuses the model's sense of who said what
[
    SystemMessage("You are a helpful assistant."),
    HumanMessage("What is PCA?"),
    SystemMessage("PCA reduces dimensions..."),   # should be AIMessage
]

# CORRECT
[
    SystemMessage("You are a helpful assistant."),
    HumanMessage("What is PCA?"),
    AIMessage("PCA reduces dimensions..."),
]
```

---

## middleware.py — Dynamic System Prompts

Swaps the agent's system prompt at runtime based on user expertise level.

**Key parts:**

- `UserRole` dataclass — carries `role` (`"expert"` / `"beginner"` / `"child"`) and `name`
- `ROLE_PROMPTS` dict — maps each role to a tailored system prompt (the only place to edit copy)
- `@dynamic_prompt` decorator — intercepts the call, injects the right system prompt, then passes the full message list to the wrapped function

```python
@dynamic_prompt
def ask_agent(messages, user_role: UserRole) -> str:
    llm = ChatOllama(model="qwen3:8b")
    return llm.invoke(messages).content
```

**To add a new role** (e.g. `"teenager"`), add one entry to `ROLE_PROMPTS` — no other changes needed:

```python
ROLE_PROMPTS["teenager"] = "You are a cool, casual assistant. Keep it short and relatable."
```

---

## dynamicModel.py — Dynamic Model Selection

Upgrades to a more capable model automatically as the conversation grows longer.

**Key parts:**

### `wrap_model_call` decorator

Mirrors the LangChain middleware pattern. Wraps a function `(request, handler) -> response`.
The `handler` does the actual LLM call. Middleware can reassign `request.model` before calling `handler(request)`.

```python
@wrap_model_call
def dynamic_model_selection(request, handler):
    message_count = len(request.state["messages"])
    request.model = advanced_model if message_count > 3 else basic_model
    return handler(request)   # handler calls the LLM with the chosen model
```

### Why `dynamic_model_selection = wrap_model_call(dynamic_model_selection)` matters

This line (what `@wrap_model_call` does behind the scenes) is important because it enforces **Single Responsibility** — each function does exactly one thing:

| Function | Responsibility |
|---|---|
| `dynamic_model_selection` | **Decision** — which model to pick based on message count |
| `handler` (inside `wrap_model_call`) | **Execution** — actually calling the LLM |

Without the decorator, `dynamic_model_selection` would have to do both:

```python
# WITHOUT decorator — decision + execution mixed together
def dynamic_model_selection(request):
    request.model = advanced_model if message_count > 3 else basic_model
    return request.model.invoke(request.messages)   # now it also owns the LLM call
```

This is tightly coupled — changing how the LLM is called (e.g. adding streaming, retries, or logging) forces you to edit `dynamic_model_selection` even though that's not its job.

With the decorator, you only ever touch one place:
- Change **decision logic** → edit `dynamic_model_selection`
- Change **how the LLM is called** → edit `wrap_model_call`

### `MemorySaver` + `thread_id`

LangGraph's built-in checkpointer. Persists the full message history keyed by `thread_id` — no manual list appending needed.

```python
agent = create_agent()   # compiles graph with MemorySaver

config = {"configurable": {"thread_id": "my-session"}}

# Each invoke automatically loads + saves history for this thread
result = agent.invoke({"messages": [HumanMessage(content="Hello")]}, config=config)
```

### Model switching logic

```
Turns 1–3 (< 4 messages in history)  →  llama3.2:3b   (fast, lightweight)
Turns 4+  (≥ 4 messages in history)  →  qwen3:8b      (more capable)
```

Change `COMPLEXITY_THRESHOLD` at the top of the file to adjust the cutover point.

---

## custAgent.py — Hooks (Observer Pattern)

### Why hooks?

Without hooks, every concern gets mixed into `agent_node`:

```python
# WITHOUT hooks — one function doing too many things
def agent_node(state):
    start = time.time()                           # timing
    log(f"history: {len(state['messages'])}")     # logging
    response = llm.invoke(messages)
    log(f"took {time.time() - start:.2f}s")       # timing again
    track_cost(response)                          # cost tracking
    return {"messages": [response]}
```

Every new concern means editing the same function — hard to maintain, impossible to turn off selectively.

With hooks, `agent_node` stays clean and each concern registers itself independently:

```python
# WITH hooks — agent_node never changes
hooks.on("before_agent", timing.start)
hooks.on("before_agent", logger.log_turn)
hooks.on("after_agent",  timing.stop)
hooks.on("after_agent",  cost_tracker.record)
```

Add or remove concerns without touching the agent at all.

### What each hook tells you in production

| Hook | What to log | What it catches |
|---|---|---|
| `before_agent` | history length, last user message | wrong context loaded, unexpected input |
| `before_model` | model name, estimated token count | wrong model selected, prompt too large |
| `after_model` | response length, empty check | silent failures, truncated replies |
| `after_agent` | total latency, slow turn alert | performance degradation, load issues |

### Troubleshooting flow

```
User reports: "the bot gave a weird answer on turn 4"
    │
    ├── before_agent  → confirm history_len=6, correct user message received
    ├── before_model  → confirm advanced model was chosen, token count normal
    ├── after_model   → raw response looks fine at this point
    └── after_agent   → latency was 8s — suspiciously slow
                              ↓
                        root cause: model was under load, not a logic bug
```

Without hooks you only see the final broken output with no trail. Hooks give you visibility into every intermediate state so you can pinpoint exactly where things went wrong.
