from abc import ABC, abstractmethod
import torch.nn as nn
from typing import List, Any
from omniid.fusion.types import FusionOutput
from omniid.objectives.types import ObjectiveMetadata, ObjectiveOutput

class BaseObjective(nn.Module, ABC):
    """
    Universal contract for all Optimization Objectives (SSL, Supervised, Hybrid).
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute_loss(self, views: List[FusionOutput], **kwargs) -> ObjectiveOutput:
        """
        Computes the primary loss scalar across arbitrary N views.
        """
        pass

    @property
    @abstractmethod
    def metadata(self) -> ObjectiveMetadata:
        """Returns the schema defining this objective's capabilities."""
        pass

    def on_step_end(self, encoder: nn.Module, *args, **kwargs):
        """
        Optional lifecycle hook for objectives that require momentum updates 
        (e.g., BYOL target networks, MoCo queues).
        """
        pass

    def validate_views(self, views: List[FusionOutput]):
        if len(views) < self.metadata.minimum_views:
            raise ValueError(f"{self.metadata.name} requires at least {self.metadata.minimum_views} views, got {len(views)}.")
