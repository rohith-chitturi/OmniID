from pydantic import BaseModel
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch

class ObjectiveMetadata(BaseModel):
    """
    Standardized schema for Objective properties, analogous to ModelMetadata.
    """
    name: str
    family: str
    requires_negatives: bool
    requires_target_network: bool
    supports_multimodal: bool
    minimum_views: int

@dataclass
class ObjectiveOutput:
    """
    The universal return payload for any Objective function.
    """
    loss: torch.Tensor
    metrics: Dict[str, float] = field(default_factory=dict)
    auxiliary_losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
