import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from omniid.fusion.base import BaseFusionModule
from omniid.fusion.registry import FUSION_REGISTRY
from omniid.fusion.types import FusionOutput

@FUSION_REGISTRY.register("transformer")
class TransformerFusion(BaseFusionModule):
    """
    Global Transformer Encoder Layer allowing all modalities to self-attend.
    We append a learnable [CLS] token to aggregate the multimodal context.
    """
    def __init__(self, target_dim: int, modality_dims: Dict[str, int], missing_strategy: str = "mask", num_heads: int = 8, num_layers: int = 2):
        super().__init__(target_dim, modality_dims, missing_strategy)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, target_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=target_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _fuse_aligned(self, aligned_tensors: Dict[str, torch.Tensor], active_modalities: list[str]) -> FusionOutput:
        # Sort keys to ensure deterministic sequence
        sorted_keys = sorted(aligned_tensors.keys())
        tensors = [aligned_tensors[k].unsqueeze(1) for k in sorted_keys]
        
        B = tensors[0].shape[0]
        seq = torch.cat(tensors, dim=1) # [B, N, target_dim]
        
        # Prepend [CLS]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls_tokens, seq], dim=1) # [B, 1+N, target_dim]
        
        # Self-Attention
        output_seq = self.transformer(seq)
        
        # The [CLS] token is the fused representation
        identity_emb = output_seq[:, 0, :]
        
        return FusionOutput(
            identity_embedding=identity_emb,
            modality_embeddings=aligned_tensors,
            active_modalities=active_modalities,
            fusion_metadata={"output_dim": self.target_dim}
        )
