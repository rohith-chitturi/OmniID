from typing import Dict, Type
from omniid.eval.base import BaseEvaluator

class EvaluatorRegistry:
    def __init__(self):
        self._registry: Dict[str, Type[BaseEvaluator]] = {}

    def register(self, name: str):
        def _register(cls):
            self._registry[name] = cls
            return cls
        return _register

    def build(self, name: str, **kwargs) -> BaseEvaluator:
        if name not in self._registry:
            raise ValueError(f"Evaluator '{name}' not found in EVALUATOR_REGISTRY.")
        return self._registry[name](**kwargs)

EVALUATOR_REGISTRY = EvaluatorRegistry()
