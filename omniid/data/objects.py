from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch

@dataclass
class IdentitySample:
    """
    A unified, strongly-typed representation of a single identity sample across multiple modalities.
    """
    identity_id: str
    vision_tensor: Optional[torch.Tensor] = None
    audio_tensor: Optional[torch.Tensor] = None
    document_tensor: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ModalityBatch:
    """
    Represents a batched collection of IdentitySamples ready for the Foundation Encoder.
    """
    vision_batch: Optional[torch.Tensor] = None
    audio_batch: Optional[torch.Tensor] = None
    document_batch: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None

@dataclass
class EmbeddingOutput:
    """
    Strongly-typed output of the representation learning heads.
    """
    universal_embedding: torch.Tensor
    modality_specific_embeddings: Optional[Dict[str, torch.Tensor]] = None
