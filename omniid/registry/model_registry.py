from typing import Dict, Type, Any
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Centralized registry for dynamically instantiating reusable neural network components.
    Ensures that components can be swapped via Hydra configurations without code changes,
    adhering to the Open/Closed Principle for Foundation Models.
    """
    _registry: Dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, name: str):
        def inner_wrapper(wrapped_class: Type[Any]):
            if name in cls._registry:
                logger.warning(f"Model {name} is already registered. Overwriting.")
            cls._registry[name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @classmethod
    def build(cls, name: str, **kwargs) -> Any:
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' not found in registry. Available: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)
