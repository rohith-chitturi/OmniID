import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List
from omniid.objectives.base import BaseObjective
from omniid.objectives.registry import OBJECTIVE_REGISTRY
from omniid.objectives.types import ObjectiveOutput, ObjectiveMetadata
from omniid.fusion.types import FusionOutput

@OBJECTIVE_REGISTRY.register("byol")
class BYOLObjective(BaseObjective):
    """
    Bootstrap Your Own Latent.
    Uses an online and target network, requires momentum updates.
    """
    def __init__(self, momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum
        # BYOL requires a predictor network on top of the online projection
        # We mock it here since actual dimension depends on fusion output
        self.predictor = None 

    def _init_predictor(self, dim: int):
        if self.predictor is None:
            self.predictor = nn.Sequential(
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
            # Move to correct device
            # This is a naive lazy-init for demonstration
            
    def compute_loss(self, views: List[FusionOutput], **kwargs) -> ObjectiveOutput:
        self.validate_views(views)
        
        # In a real trainer, view 0 is from online network, view 1 is from target network
        online_view = views[0].identity_embedding
        target_view = views[1].identity_embedding
        
        self._init_predictor(online_view.size(-1))
        self.predictor.to(online_view.device)
        
        # Online prediction
        p = self.predictor(online_view)
        
        # Target representation
        z = target_view.detach()
        
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        
        # Cosine similarity loss (negated)
        loss = 2 - 2 * (p * z).sum(dim=-1).mean()
        
        return ObjectiveOutput(loss=loss)

    def on_step_end(self, online_encoder: nn.Module, target_encoder: nn.Module, **kwargs):
        """
        Lifecycle hook to execute Exponential Moving Average (EMA) update on target_encoder.
        """
        for online_params, target_params in zip(online_encoder.parameters(), target_encoder.parameters()):
            target_params.data = self.momentum * target_params.data + (1 - self.momentum) * online_params.data

    @property
    def metadata(self) -> ObjectiveMetadata:
        return ObjectiveMetadata(
            name="byol",
            family="distillation",
            requires_negatives=False,
            requires_target_network=True,
            supports_multimodal=False,
            minimum_views=2
        )
