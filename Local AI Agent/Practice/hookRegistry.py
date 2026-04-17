from dataclasses import dataclass
from functools import wraps
from collections import defaultdict
import time
from typing import Callable, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langgraph.graph import MessagesState
from langsmith import traceable

# ---------------------------------------------------------------------------
# 1. ModelRequest
# model is typed as Runnable (not ChatOllama)
# RunnableBinding, which is a Runnable but not a ChatOllama.
# ---------------------------------------------------------------------------

@dataclass
class ModelRequest:
    model: Runnable
    messages: Sequence[BaseMessage] # input to the model chain; same as C# generics
    state: MessagesState
    
ModelResponse = BaseMessage

# ---------------------------------------------------------------------------
# 2. HookRegistry — Observer Pattern
# ---------------------------------------------------------------------------

class HookRegistry:
    def __init__(self):
        self._hooks: dict[str, list[Callable]] = defaultdict(list)
        self._timing: dict = {}
        self._register_defaults()
        
    def _register_defaults(self):
        self.on("before_agent", self.on_before_agent)
        self.on("before_model", self.on_before_model)
        self.on("after_model",  self.on_after_model)
        self.on("after_agent",  self.on_after_agent)
        self.on("on_error",     self.on_error)

    def on(self, event: str, callback: Callable) -> None:
        self._hooks[event].append(callback)

    def emit(self, event: str, *args) -> None:
        for callback in self._hooks[event]:
            callback(*args)

    # ---------------------------------------------------------------------------
    # 3. Timing + exception callbacks
    # Smith has cloud version only of traceable
    # ---------------------------------------------------------------------------

    @traceable(name="hook.before_agent")
    def on_before_agent(self, state: MessagesState) -> dict:
        self._timing["agent_start"] = time.time()
        return {"history_length": len(state["messages"])}

    @traceable(name="hook.before_model")
    def on_before_model(self, request: ModelRequest) -> dict:
        self._timing["model_start"] = time.time()
        # getattr safely handles both ChatOllama (.model attr) and RunnableBinding
        return {"model": getattr(request.model, 'model', str(request.model))}

    @traceable(name="hook.after_model")
    def on_after_model(self, _response: ModelResponse) -> dict:
        elapsed = time.time() - self._timing["model_start"]
        return {"elapsed_seconds": round(elapsed, 2)}

    @traceable(name="hook.after_agent")
    def on_after_agent(self, _state: MessagesState) -> dict:
        elapsed = time.time() - self._timing["agent_start"]
        return {"total_seconds": round(elapsed, 2)}

    @traceable(name="hook.on_error")
    def on_error(self, exc: Exception, request: ModelRequest) -> dict:
        start = self._timing.get("model_start") or self._timing.get("agent_start", time.time())
        elapsed = time.time() - start
        return {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "model": getattr(request.model, 'model', str(request.model)),
            "elapsed_seconds": round(elapsed, 2),
        }