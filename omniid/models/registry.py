from typing import Dict, Type
from omniid.models.base import BaseFoundationEncoder

class BackboneRegistry:
    def __init__(self):
        self._registry: Dict[str, Type[BaseFoundationEncoder]] = {}

    def register(self, name: str):
        def _register(cls):
            self._registry[name] = cls
            return cls
        return _register

    def build(self, name: str, **kwargs) -> BaseFoundationEncoder:
        if name not in self._registry:
            raise ValueError(f"Encoder '{name}' not found in BACKBONE_REGISTRY.")
        return self._registry[name](**kwargs)

BACKBONE_REGISTRY = BackboneRegistry()
