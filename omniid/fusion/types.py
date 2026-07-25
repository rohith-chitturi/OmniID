from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch

@dataclass
class ModalityEmbedding:
    """
    Standardized input representation for the Fusion module.
    Decouples fusion algorithms from specific encoder types.
    """
    name: str
    modality_type: str
    embedding: torch.Tensor
    mask: Optional[torch.Tensor] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_present: bool = True

@dataclass
class FusionOutput:
    """
    Standardized output for multimodal fusion strategies.
    Consumed by downstream self-supervised objectives.
    """
    identity_embedding: torch.Tensor
    modality_embeddings: Dict[str, torch.Tensor]
    active_modalities: list[str]
    attention_maps: Optional[torch.Tensor] = None
    confidence_score: Optional[torch.Tensor] = None
    fusion_metadata: Dict[str, Any] = field(default_factory=dict)
    projection_metadata: Dict[str, Any] = field(default_factory=dict)
