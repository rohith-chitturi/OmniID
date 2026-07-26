from typing import Dict, Any, List
from omniid.runtime.types import InferenceRequest
from omniid.model_zoo.types import ModelManifest

class RuntimeValidator:
    """
    Control Plane component for request validation before execution.
    """
    def __init__(self, manifest: ModelManifest):
        self.manifest = manifest

    def validate_request(self, request: InferenceRequest) -> None:
        """
        Ensures the incoming request is compatible with the loaded model.
        Raises ValueError if validation fails.
        """
        # 1. Modality Compatibility
        expected_modalities = ["vision"]
        if self.manifest.fusion:
            # Simplified mock logic: if fused, it expects face + document
            expected_modalities = ["face", "document"]
            
        for modality in request.modalities:
            if modality not in expected_modalities:
                raise ValueError(f"Model '{self.manifest.model_name}' does not support modality: {modality}")
                
        # 2. Input existence
        if not request.inputs:
            raise ValueError("Inference request contains no inputs.")
            
        # 3. Size constraints (e.g. max batch limits or image dims)
        # Assuming inputs values are lists representing batch dimension
        for mod, data in request.inputs.items():
            if isinstance(data, list) and len(data) > 32:
                raise ValueError(f"Batch size {len(data)} exceeds maximum allowed (32)")
