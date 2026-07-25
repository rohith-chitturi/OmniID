from typing import Dict, Type
from omniid.objectives.base import BaseObjective

class ObjectiveRegistry:
    def __init__(self):
        self._registry: Dict[str, Type[BaseObjective]] = {}

    def register(self, name: str):
        def _register(cls):
            self._registry[name] = cls
            return cls
        return _register

    def build(self, name: str, **kwargs) -> BaseObjective:
        if name not in self._registry:
            raise ValueError(f"Objective strategy '{name}' not found in OBJECTIVE_REGISTRY.")
        return self._registry[name](**kwargs)

OBJECTIVE_REGISTRY = ObjectiveRegistry()
