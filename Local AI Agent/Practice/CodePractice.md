# LangGraph Practice Plan

## Goal
Build real understanding by rebuilding key parts step-by-step.
Try from memory first. Peek only when stuck > 10 min.

---

## Practice 1 — Basic Graph Flow (Easy)
**File:** `dynamicModel.py`
**Learn:** How LangGraph connects nodes into a runnable graph
**What to remove:** Everything except the bare StateGraph skeleton
**What to rebuild:**
- `StateGraph(MessagesState)` with a single `agent_node`
- `add_node`, `add_edge(START, ...)`, `add_edge(..., END)`
- `graph.compile(checkpointer=MemorySaver())`
- Inside `agent_node`: call `factory.create(0).invoke(messages)` directly (no middleware)

**Key concept:** LangGraph is just a state machine — nodes transform state, edges define flow

---

## Practice 2 — Model Switching (Medium)
**File:** `dynamicModel.py` + `modelFactory.py`
**Learn:** How turn count drives model selection
**What to remove:** `dynamic_model_selection` and `ModelFactory.create()` logic
**What to rebuild:**
- Count `HumanMessage` turns in `state["messages"]`
- Return `ADVANCED_MODEL` if turns > threshold, else `BASIC_MODEL`
- Wrap in `ModelFactory.create(turns, has_tools)`

**Key concept:** Factory Pattern — callers just ask for a model, factory decides which one

---

## Practice 3 — Middleware Chain (Hard)
**File:** `dynamicModel.py`
**Learn:** How Chain of Responsibility passes a request through layers
**What to remove:** `wrap_model_call`, `run_middleware_chain`, `call_next` recursion
**What to rebuild:**
```python
def run_middleware_chain(request, middleware):
    def call_next(i, req):
        if i == len(middleware):
            return req.model.invoke(req.messages)  # base case — call LLM
        return middleware[i](req, lambda r: call_next(i + 1, r))
    return call_next(0, request)
```
**Key concept:** Each middleware calls `handler(request)` to pass control forward — never calls LLM directly

---

## Practice 4 — Full Rebuild (Max 10 min)
**Files:** `dynamicModel.py`, `modelFactory.py`, `hookRegistry.py`, `agentBuilder.py`
**Learn:** Whether you can hold all 4 patterns in your head at once
**What to rebuild from scratch:** Graph + middleware chain + model factory + observer hooks
**Rule:** If stuck > 10 min on one part → peek → continue. Time yourself.

---

## Practice 5 — Tool Binding + Model Upgrade (Medium)
**File:** `dynamicModel.py`, `modelFactory.py`
**Learn:** Why tools require a stronger model and change the model's type
**What to remove:**
- `has_tools=bool(_tools)` from `agent_node`'s `factory.create()` call
- `model.bind_tools(_tools)` from `_run_node`

**What to rebuild:**
- Add `has_tools: bool` param to `ModelFactory.create()` → force `ADVANCED_MODEL` when True
- Call `model.bind_tools(_tools)` in `_run_node` before building `ModelRequest`

**Key concept:** `bind_tools()` returns a `RunnableBinding`, not a `ChatOllama` — the type changes. Small models (3b) lack reliable tool-calling, so `has_tools=True` forces the upgrade.

---

## Practice 6 — ReAct Loop: ToolNode + Router (Hard)
**File:** `dynamicModel.py`
**Learn:** Why tool calls need a separate node to actually execute
**What to remove:**
- `ToolNode` import and `graph.add_node("tools", ToolNode(_tools))`
- `should_continue` function
- `add_conditional_edges` and `add_edge("tools", "agent")`

**What to rebuild:**
```python
from langgraph.prebuilt import ToolNode

def should_continue(state):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

graph.add_node("tools", ToolNode(_tools))
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
```
**Verify:** "What's the weather where I am?" → `locate_user` runs → `get_weather` runs → answer

**Key concept:** The model returns a *tool_call message* — it does NOT run the tool. `ToolNode` is what executes it. Without the loop, tool calls just sit in state forever.

---

## Practice 7 — Middleware Guard: Protect RunnableBinding (Tricky)
**File:** `dynamicModel.py`
**Learn:** Why middleware order matters and how to protect work done upstream
**What to remove:** The `isinstance(request.model, RunnableBinding)` guard — make middleware always override
**Observe:** Weather question breaks — agent says "I don't have location access"
**What to restore:**
```python
from langchain_core.runnables.base import RunnableBinding
if not isinstance(request.model, RunnableBinding):
    request.model = factory.create(human_turns)
```
**Key concept:** `agent_node` binds tools first, then middleware runs. Without the guard, middleware replaces the bound model with a bare `ChatOllama` — tools are silently lost. Always check what you're overwriting.

---

## Process Flow (Full)

```
[LangGraph]  agent.invoke()                         ← compiled graph entry point
[LangGraph]    → agent_node                         ← node function in StateGraph
[custom]           → factory.create(turns, has_tools=True) → qwen3:8b
[LangChain]        → model.bind_tools([locate_user, get_weather])  ← returns RunnableBinding
[custom]           → middleware chain (isinstance guard skips override)
[LangChain]        → model.invoke(messages)         ← Runnable interface, returns tool_call msg
[LangGraph]    → should_continue() → "tools"        ← conditional edge router
[LangGraph]    → ToolNode executes locate_user / get_weather  ← prebuilt tool executor
[LangGraph]    → agent_node again with tool result  ← loop back edge
[LangChain]        → model.invoke(messages)         ← returns final answer
[LangGraph]    → should_continue() → END
[LangGraph]    → MemorySaver saves state            ← checkpointer
```

**One-line summary: LangChain = talking to the model. LangGraph = controlling the flow between steps.**

## Implementation sequence from scratch:

```
1. loadEnv.py          — no deps; defines all config constants
2. hookRegistry.py     — defines ModelRequest / ModelResponse dataclasses + HookRegistry (Observer)
3. modelFactory.py     — depends on loadEnv; pure logic, easy to test alone
4. modelMiddleware.py  — depends on hookRegistry + modelFactory; Chain of Responsibility
5. dynamicModel.py     — depends on all above; wires the LangGraph
6. agentBuilder.py     — depends on hookRegistry + dynamicModel; Builder wrapper on top
7. weather.py          — standalone tool, add whenever needed
```