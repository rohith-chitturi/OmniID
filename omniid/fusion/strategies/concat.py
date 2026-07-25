import torch
import torch.nn as nn
from typing import Dict
from omniid.fusion.base import BaseFusionModule
from omniid.fusion.registry import FUSION_REGISTRY
from omniid.fusion.types import FusionOutput

@FUSION_REGISTRY.register("concat")
class ConcatFusion(BaseFusionModule):
    """
    Concatenates aligned embeddings channel-wise.
    """
    def __init__(self, target_dim: int, modality_dims: Dict[str, int], missing_strategy: str = "mask"):
        super().__init__(target_dim, modality_dims, missing_strategy)
        # We concatenate N modalities of size target_dim
        self.num_modalities = len(modality_dims)
        self.output_dim = self.num_modalities * target_dim
        
    def _fuse_aligned(self, aligned_tensors: Dict[str, torch.Tensor], active_modalities: list[str]) -> FusionOutput:
        # Sort by key to guarantee deterministic concatenation order
        sorted_keys = sorted(aligned_tensors.keys())
        tensors = [aligned_tensors[k] for k in sorted_keys]
        
        identity_emb = torch.cat(tensors, dim=-1) # [B, N * target_dim]
        
        return FusionOutput(
            identity_embedding=identity_emb,
            modality_embeddings=aligned_tensors,
            active_modalities=active_modalities,
            fusion_metadata={"output_dim": self.output_dim}
        )
