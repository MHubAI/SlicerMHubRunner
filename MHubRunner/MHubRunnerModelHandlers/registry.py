from .base import GenericModelHandler, ModelHandler
from .handlers import Grt123LungCancerHandler


class ModelHandlerRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, type[ModelHandler]] = {}
        self.register(Grt123LungCancerHandler)

    def register(self, handler_type: type[ModelHandler]) -> None:
        for model_name in handler_type.model_names:
            if model_name in self._handlers:
                raise ValueError(f"An output handler is already registered for {model_name!r}.")
            self._handlers[model_name] = handler_type

    def handler_for(self, model_name: str) -> ModelHandler:
        handler_type = self._handlers.get(model_name, GenericModelHandler)
        return handler_type()
