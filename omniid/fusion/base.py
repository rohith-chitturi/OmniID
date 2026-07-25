from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Dict, Any, Optional
from omniid.fusion.types import ModalityEmbedding, FusionOutput
from omniid.fusion.alignment import EmbeddingAlignmentLayer

class BaseFusionModule(nn.Module, ABC):
    """
    Universal contract for all Multimodal Fusion Strategies.
    Consumes Standardized ModalityEmbeddings and delegates projection to EmbeddingAlignmentLayer.
    """
    def __init__(self, target_dim: int, modality_dims: Dict[str, int], missing_strategy: str = "mask"):
        super().__init__()
        self.target_dim = target_dim
        self.missing_strategy = missing_strategy
        self.alignment_layer = EmbeddingAlignmentLayer(target_dim, modality_dims)

    def fuse(self, modalities: Dict[str, ModalityEmbedding]) -> FusionOutput:
        """
        Public contract for fusion.
        Handles generic alignment, then delegates to specific math.
        """
        aligned_tensors = self.alignment_layer(modalities, missing_strategy=self.missing_strategy)
        active = [name for name, m in modalities.items() if m.is_present]
        return self._fuse_aligned(aligned_tensors, active)

    @abstractmethod
    def _fuse_aligned(self, aligned_tensors: Dict[str, nn.Module], active_modalities: list[str]) -> FusionOutput:
        """
        The specific fusion algorithm (e.g., Concat, Attention) to be implemented by subclasses.
        """
        pass

    def visualize_attention(self, output: FusionOutput) -> Any:
        """
        Placeholder API for explainability.
        """
        if output.attention_maps is None:
            raise NotImplementedError("This fusion strategy does not produce attention maps.")
        return output.attention_maps
