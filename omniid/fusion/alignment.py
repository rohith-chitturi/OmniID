import torch
import torch.nn as nn
from typing import Dict, Any
from omniid.fusion.types import ModalityEmbedding

class EmbeddingAlignmentLayer(nn.Module):
    """
    Normalizes and projects heterogeneous embeddings into a shared Latent Space.
    Abstracts mismatching dimensions (e.g. 768 vs 1024) away from the Fusion algorithms.
    """
    def __init__(self, target_dim: int, modality_dims: Dict[str, int]):
        super().__init__()
        self.target_dim = target_dim
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, target_dim) if dim != target_dim else nn.Identity()
            for name, dim in modality_dims.items()
        })
        self.missing_modality_token = nn.Parameter(torch.zeros(1, target_dim))
        nn.init.normal_(self.missing_modality_token, std=0.02)

    def forward(self, modalities: Dict[str, ModalityEmbedding], missing_strategy: str = "mask") -> Dict[str, torch.Tensor]:
        aligned = {}
        
        for name, mod in modalities.items():
            if name not in self.projections:
                raise ValueError(f"Modality '{name}' not registered in alignment layer.")
                
            if mod.is_present:
                # Project to common latent space
                projected = self.projections[name](mod.embedding)
                aligned[name] = projected
            else:
                # Handle missing modality dynamically
                B = next(iter(modalities.values())).embedding.size(0)
                if missing_strategy == "mask" or missing_strategy == "zero":
                    aligned[name] = torch.zeros(B, self.target_dim, device=mod.embedding.device)
                elif missing_strategy == "learned_token":
                    aligned[name] = self.missing_modality_token.expand(B, -1)
                else:
                    raise ValueError(f"Unknown missing_strategy: {missing_strategy}")
                    
        return aligned
