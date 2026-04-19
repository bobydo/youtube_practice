from dataclasses import dataclass
from functools import wraps
from collections import defaultdict
import time
from typing import Callable, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langgraph.graph import MessagesState
from loggerSetup import get_logger

logger = get_logger(__name__)

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

    def on_before_agent(self, state: MessagesState) -> None:
        self._timing["agent_start"] = time.time()
        logger.info("before_agent  history_length=%d", len(state["messages"]))

    def on_before_model(self, request: ModelRequest) -> None:
        self._timing["model_start"] = time.time()
        model_name = getattr(request.model, 'model', str(request.model))
        logger.info("before_model  model=%s", model_name)

    def on_after_model(self, _response: ModelResponse) -> None:
        elapsed = time.time() - self._timing["model_start"]
        preview_response = str(_response.content)[:800]
        logger.info("after_model   elapsed=%.2fs preview_response=%s", elapsed, preview_response)

    def on_after_agent(self, _state: MessagesState) -> None:
        elapsed = time.time() - self._timing["agent_start"]
        preview_state = str(_state["messages"][-1].content)[:80]
        logger.info("after_agent   total=%.2fs preview_state=%s", elapsed, preview_state)

    def on_error(self, exc: Exception, request: ModelRequest) -> None:
        start = self._timing.get("model_start") or self._timing.get("agent_start", time.time())
        elapsed = time.time() - start
        model_name = getattr(request.model, 'model', str(request.model))
        logger.error("on_error  %s: %s  model=%s  elapsed=%.2fs",
            type(exc).__name__, exc, model_name, elapsed)