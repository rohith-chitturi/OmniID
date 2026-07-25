import torch
import torch.nn as nn
from typing import Dict
from omniid.fusion.base import BaseFusionModule
from omniid.fusion.registry import FUSION_REGISTRY
from omniid.fusion.types import FusionOutput

@FUSION_REGISTRY.register("weighted_sum")
class WeightedSumFusion(BaseFusionModule):
    """
    Learnable weighted sum of all modality embeddings.
    Maintains the original target_dim.
    """
    def __init__(self, target_dim: int, modality_dims: Dict[str, int], missing_strategy: str = "mask"):
        super().__init__(target_dim, modality_dims, missing_strategy)
        # Learnable weights per modality
        self.weights = nn.ParameterDict({
            name: nn.Parameter(torch.ones(1)) for name in modality_dims.keys()
        })
        
    def _fuse_aligned(self, aligned_tensors: Dict[str, torch.Tensor], active_modalities: list[str]) -> FusionOutput:
        identity_emb = 0
        for name, tensor in aligned_tensors.items():
            # Apply learned weight
            identity_emb = identity_emb + (tensor * self.weights[name])
            
        return FusionOutput(
            identity_embedding=identity_emb,
            modality_embeddings=aligned_tensors,
            active_modalities=active_modalities,
            fusion_metadata={"output_dim": self.target_dim}
        )
