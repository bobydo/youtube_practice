from typing import Callable
from langchain_core.messages import HumanMessage
from langsmith import traceable
from hookRegistry import ModelRequest, ModelResponse
from modelFactory import ModelFactory
from loadEnv import MAX_RETRIES

_factory = ModelFactory()


class ModelMiddleware:
    @staticmethod
    def wrap_model_call(fn: Callable) -> Callable:
        def wrapper(request: ModelRequest, handler: Callable) -> ModelResponse:
            return fn(request, handler)
        return wrapper

    @staticmethod
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

    @staticmethod
    def dynamic_model_selection(request: ModelRequest, handler: Callable) -> ModelResponse:
        from langchain_core.runnables.base import RunnableBinding
        if not isinstance(request.model, RunnableBinding):
            human_turns = sum(isinstance(m, HumanMessage) for m in request.state["messages"])
            request.model = _factory.create(human_turns)
        return handler(request)


# @staticmethod and @wrap_model_call can't stack directly
ModelMiddleware.dynamic_model_selection = ModelMiddleware.wrap_model_call(
    ModelMiddleware.dynamic_model_selection
)
