from typing import Dict, Type
from omniid.runtime.exporters.base import BaseExporter

class ExporterRegistry:
    def __init__(self):
        self._registry: Dict[str, Type[BaseExporter]] = {}

    def register(self, name: str):
        def _register(cls):
            self._registry[name] = cls
            return cls
        return _register

    def build(self, name: str, **kwargs) -> BaseExporter:
        # Trigger registrations
        import omniid.runtime.exporters.onnx
        import omniid.runtime.exporters.torchscript
        
        if name not in self._registry:
            raise ValueError(f"Exporter '{name}' not found in EXPORTER_REGISTRY.")
        return self._registry[name](**kwargs)

EXPORTER_REGISTRY = ExporterRegistry()
