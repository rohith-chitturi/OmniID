from typing import Dict, Any, List
from omniid.model_zoo.types import ModelManifest, ModelCard
from omniid.model_zoo.registry import ZOO_REGISTRY
from omniid.model_zoo.resolver import CheckpointResolver
from omniid.model_zoo.compatibility import CompatibilityValidator

class ModelZoo:
    """
    Facade for discovering, validating, and loading published models.
    """
    def __init__(self):
        self.resolver = CheckpointResolver()
        self.validator = CompatibilityValidator()

    def list_models(self) -> List[str]:
        """
        List all models discovered by the configured resolver providers.
        """
        # In a full implementation, we'd query providers.
        # For this demonstration, we return the in-memory cache.
        return list(ZOO_REGISTRY.list_models().keys())

    def get_model_info(self, model_name: str, provider: str = "local") -> ModelManifest:
        """
        Retrieves the manifest for a given model.
        """
        try:
            return ZOO_REGISTRY.get(model_name)
        except KeyError:
            # Try to resolve it
            manifest_dict, _ = self.resolver.resolve(model_name, provider_preference=provider)
            manifest = ModelManifest(**manifest_dict)
            ZOO_REGISTRY.register(manifest)
            return manifest

    def load(self, model_name: str, current_config: Dict[str, Any], provider: str = "local") -> str:
        """
        Validates compatibility and returns the local path to the checkpoint weights.
        """
        # 1. Resolve Manifest and Path
        manifest_dict, ckpt_path = self.resolver.resolve(model_name, provider_preference=provider)
        manifest = ModelManifest(**manifest_dict)
        ZOO_REGISTRY.register(manifest)
        
        # 2. Validate Compatibility
        is_compatible, msg = self.validator.validate(manifest, current_config)
        if not is_compatible:
            raise RuntimeError(f"Compatibility Validation Failed for {model_name}: {msg}")
            
        # 3. Return path to weights
        # The execution engine or WeightManager will handle torch.load(ckpt_path)
        return ckpt_path

    def get_model_card(self, model_name: str, provider: str = "local") -> ModelCard:
        """
        Returns the human-readable ModelCard (requires manifest + extra metadata).
        """
        # In a real implementation, this would fetch a model_card.json or README.
        manifest = self.get_model_info(model_name, provider)
        return ModelCard(
            manifest=manifest,
            intended_use="Universal identity representation.",
            limitations="Not intended for generative tasks.",
            datasets_used=[manifest.dataset_fingerprint],
            supported_modalities=["face", "document"] if manifest.fusion else ["vision"]
        )
