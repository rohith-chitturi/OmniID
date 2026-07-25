import torch
import torch.nn.functional as F
from typing import List
from omniid.objectives.base import BaseObjective
from omniid.objectives.registry import OBJECTIVE_REGISTRY
from omniid.objectives.types import ObjectiveOutput, ObjectiveMetadata
from omniid.fusion.types import FusionOutput

@OBJECTIVE_REGISTRY.register("simclr")
class SimCLRObjective(BaseObjective):
    """
    Contrastive Learning via InfoNCE Loss.
    Pulls augmented views together, pushes negatives apart.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def compute_loss(self, views: List[FusionOutput], **kwargs) -> ObjectiveOutput:
        self.validate_views(views)
        
        # SimCLR typically expects 2 augmented views
        z_i = F.normalize(views[0].identity_embedding, dim=-1)
        z_j = F.normalize(views[1].identity_embedding, dim=-1)
        
        batch_size = z_i.size(0)
        
        # Concatenate for self-similarity matrix
        z = torch.cat([z_i, z_j], dim=0) # [2N, D]
        sim_matrix = torch.exp(torch.mm(z, z.t().contiguous()) / self.temperature) # [2N, 2N]
        
        # Mask out self-similarity (the diagonal)
        mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=z.device)).float()
        sim_matrix = sim_matrix * mask
        
        # Compute InfoNCE
        # Positives are at distance N in the concatenated matrix
        positives = torch.cat([
            torch.diag(sim_matrix, batch_size),
            torch.diag(sim_matrix, -batch_size)
        ], dim=0)
        
        nominator = positives
        denominator = sim_matrix.sum(dim=-1)
        
        loss = -torch.log(nominator / denominator).mean()
        
        metrics = {
            "contrastive_accuracy": 0.0 # Mock metric
        }
        
        return ObjectiveOutput(loss=loss, metrics=metrics)

    @property
    def metadata(self) -> ObjectiveMetadata:
        return ObjectiveMetadata(
            name="simclr",
            family="contrastive",
            requires_negatives=True,
            requires_target_network=False,
            supports_multimodal=False,
            minimum_views=2
        )
