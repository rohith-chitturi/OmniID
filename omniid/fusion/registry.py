from typing import Dict, Type
from omniid.fusion.base import BaseFusionModule

class FusionRegistry:
    def __init__(self):
        self._registry: Dict[str, Type[BaseFusionModule]] = {}

    def register(self, name: str):
        def _register(cls):
            self._registry[name] = cls
            return cls
        return _register

    def build(self, name: str, **kwargs) -> BaseFusionModule:
        if name not in self._registry:
            raise ValueError(f"Fusion strategy '{name}' not found in FUSION_REGISTRY.")
        return self._registry[name](**kwargs)

FUSION_REGISTRY = FusionRegistry()
