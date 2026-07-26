from abc import ABC, abstractmethod
from typing import Dict, Any, Type
import os
import hashlib

class BaseProvider(ABC):
    """
    Abstract Storage Backend for resolving and downloading checkpoints.
    """
    @abstractmethod
    def resolve_manifest(self, model_name: str) -> Dict[str, Any]:
        """Fetch the JSON manifest representing the model."""
        pass
        
    @abstractmethod
    def fetch_checkpoint(self, checkpoint_hash: str) -> str:
        """Download (if necessary) and return the local filepath to the checkpoint."""
        pass

class LocalProvider(BaseProvider):
    """
    Resolves checkpoints from a local directory.
    """
    def __init__(self, zoo_dir: str = "./zoo"):
        self.zoo_dir = zoo_dir
        os.makedirs(self.zoo_dir, exist_ok=True)
        
    def resolve_manifest(self, model_name: str) -> Dict[str, Any]:
        manifest_path = os.path.join(self.zoo_dir, f"{model_name}.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Model manifest not found in local zoo: {manifest_path}")
            
        import json
        with open(manifest_path, "r") as f:
            return json.load(f)
            
    def fetch_checkpoint(self, checkpoint_hash: str) -> str:
        # In a real local provider, we might map hash to filename, or the manifest would just have the relative path.
        # Here we assume the checkpoint is stored as {hash}.pth
        ckpt_path = os.path.join(self.zoo_dir, f"{checkpoint_hash}.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_hash} not found locally at {ckpt_path}")
        return ckpt_path

class CheckpointResolver:
    """
    Orchestrates the resolution of checkpoints across multiple providers.
    """
    def __init__(self):
        self.providers: Dict[str, BaseProvider] = {
            "local": LocalProvider()
        }
        
    def register_provider(self, name: str, provider: BaseProvider):
        self.providers[name] = provider
        
    def resolve(self, model_name: str, provider_preference: str = "local") -> tuple[Dict[str, Any], str]:
        """
        Returns (manifest_dict, local_checkpoint_path)
        """
        if provider_preference not in self.providers:
            raise ValueError(f"Unknown provider '{provider_preference}'")
            
        provider = self.providers[provider_preference]
        manifest_dict = provider.resolve_manifest(model_name)
        ckpt_path = provider.fetch_checkpoint(manifest_dict["checkpoint_hash"])
        
        return manifest_dict, ckpt_path
