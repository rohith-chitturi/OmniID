import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from omniid.fusion.base import BaseFusionModule
from omniid.fusion.registry import FUSION_REGISTRY
from omniid.fusion.types import FusionOutput

@FUSION_REGISTRY.register("cross_attention")
class CrossAttentionFusion(BaseFusionModule):
    """
    Standard query-key-value attention mechanism.
    Typically, a primary modality (e.g. Vision) queries the secondary modalities.
    """
    def __init__(self, target_dim: int, modality_dims: Dict[str, int], missing_strategy: str = "mask", num_heads: int = 8, query_modality: str = "face"):
        super().__init__(target_dim, modality_dims, missing_strategy)
        self.num_heads = num_heads
        self.query_modality = query_modality
        
        if self.query_modality not in modality_dims:
            raise ValueError(f"query_modality '{self.query_modality}' not found in provided modalities.")
            
        self.attention = nn.MultiheadAttention(embed_dim=target_dim, num_heads=num_heads, batch_first=True)

    def _fuse_aligned(self, aligned_tensors: Dict[str, torch.Tensor], active_modalities: list[str]) -> FusionOutput:
        # Query
        query = aligned_tensors[self.query_modality].unsqueeze(1) # [B, 1, target_dim]
        
        # Keys & Values (all other modalities)
        kv_tensors = []
        for name, tensor in aligned_tensors.items():
            if name != self.query_modality:
                kv_tensors.append(tensor.unsqueeze(1))
                
        if not kv_tensors:
            # If no other modalities exist, fallback to identity
            return FusionOutput(
                identity_embedding=aligned_tensors[self.query_modality],
                modality_embeddings=aligned_tensors,
                active_modalities=active_modalities,
                fusion_metadata={"output_dim": self.target_dim, "attention_used": False}
            )
            
        kv = torch.cat(kv_tensors, dim=1) # [B, N-1, target_dim]
        
        # Apply Cross Attention
        attn_output, attn_weights = self.attention(query, kv, kv)
        
        identity_emb = attn_output.squeeze(1) # [B, target_dim]
        
        return FusionOutput(
            identity_embedding=identity_emb,
            modality_embeddings=aligned_tensors,
            active_modalities=active_modalities,
            attention_maps=attn_weights,
            fusion_metadata={"output_dim": self.target_dim, "attention_used": True}
        )
