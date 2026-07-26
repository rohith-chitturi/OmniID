from typing import Dict
from omniid.model_zoo.types import ModelManifest

class ZooRegistry:
    """
    In-memory cache/registry of loaded or discovered model manifests.
    """
    def __init__(self):
        self._manifests: Dict[str, ModelManifest] = {}

    def register(self, manifest: ModelManifest):
        self._manifests[manifest.model_name] = manifest

    def get(self, model_name: str) -> ModelManifest:
        if model_name not in self._manifests:
            raise KeyError(f"Model '{model_name}' not found in ZooRegistry.")
        return self._manifests[model_name]
        
    def list_models(self) -> Dict[str, ModelManifest]:
        return self._manifests

ZOO_REGISTRY = ZooRegistry()
