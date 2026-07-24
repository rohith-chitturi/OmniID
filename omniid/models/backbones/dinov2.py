import torch
import torch.nn as nn
from typing import Dict, Any
from omniid.models.base import BaseFoundationEncoder
from omniid.models.registry import BACKBONE_REGISTRY
from omniid.models.weights import WeightManager

@BACKBONE_REGISTRY.register("dinov2")
class DINOv2Encoder(BaseFoundationEncoder):
    def __init__(self, size: str = "vits14"):
        super().__init__()
        self.size = size
        
        # In a real implementation, we would load the actual PyTorch hub model:
        # self.model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{size}')
        # Using a mock linear layer to simulate embedding extraction for the framework:
        self.mock_model = nn.Linear(3 * 224 * 224, self.embedding_dim)

    def encode(self, image: torch.Tensor, mode: str = "cls") -> torch.Tensor:
        if mode not in ["cls", "patch", "pooled"]:
            raise ValueError(f"Invalid extraction mode: {mode}")
            
        # Mocking the forward pass
        B = image.shape[0]
        flat = image.view(B, -1)
        embeds = self.mock_model(flat) # [B, embedding_dim]
        
        if mode == "cls" or mode == "pooled":
            return embeds
        elif mode == "patch":
            # Mock patch outputs [B, num_patches, embedding_dim]
            num_patches = (self.input_resolution // self.patch_size) ** 2
            return embeds.unsqueeze(1).repeat(1, num_patches, 1)

    def get_preprocessing_transforms(self) -> Dict[str, Any]:
        return {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "resolution": self.input_resolution
        }

    @property
    def embedding_dim(self) -> int:
        dims = {"vits14": 384, "vitb14": 768, "vitl14": 1024, "vitg14": 1536}
        return dims.get(self.size, 384)

    @property
    def patch_size(self) -> int:
        return 14

    @property
    def input_resolution(self) -> int:
        return 224
