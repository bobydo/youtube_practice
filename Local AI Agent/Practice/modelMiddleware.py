import time
from typing import Callable
from langchain_core.messages import HumanMessage
from hookRegistry import ModelRequest, ModelResponse
from modelFactory import ModelFactory
from loadEnv import MAX_RETRIES
from loggerSetup import get_logger

logger = get_logger(__name__)


class ModelMiddleware:
    def __init__(self, factory: ModelFactory | None = None):
        self._factory = factory or ModelFactory()

    def _wrap(self, fn: Callable) -> Callable:
        def wrapper(request: ModelRequest, handler: Callable) -> ModelResponse:
            return fn(request, handler)
        return wrapper

    async def run_middleware_chain(self, request: ModelRequest, middleware: list[Callable]) -> ModelResponse:
        t0 = time.time()
        logger.info("middleware_chain  START  steps=%d", len(middleware))

        result = await self._call_next(0, request, middleware)
        logger.info("middleware_chain  END    elapsed=%.2fs", time.time() - t0)
        return result

    async def dynamic_model_selection(self, request: ModelRequest, handler: Callable) -> ModelResponse:
        from langchain_core.runnables.base import RunnableBinding
        if not isinstance(request.model, RunnableBinding):
            human_turns = sum(isinstance(m, HumanMessage) for m in request.state["messages"])
            request.model = self._factory.create(human_turns)
        return await handler(request)

    async def _call_next(self, i: int, req: ModelRequest, middleware: list[Callable]) -> ModelResponse:
        if i == len(middleware):
            for attempt in range(MAX_RETRIES):
                try:
                    return await req.model.ainvoke(req.messages)
                except Exception:
                    if attempt < MAX_RETRIES - 1:
                        pass
                    else:
                        raise
        return await middleware[i](req, lambda r: self._call_next(i + 1, r, middleware))
