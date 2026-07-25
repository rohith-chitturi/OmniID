import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List
from omniid.objectives.base import BaseObjective
from omniid.objectives.registry import OBJECTIVE_REGISTRY
from omniid.objectives.types import ObjectiveOutput, ObjectiveMetadata
from omniid.fusion.types import FusionOutput

@OBJECTIVE_REGISTRY.register("clip")
class CLIPLoss(BaseObjective):
    """
    Standard Cross-Modal Contrastive Loss.
    Matches two different modalities (e.g., Face vs Document) symmetrically.
    """
    def __init__(self, initial_temperature: float = 0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / initial_temperature)))

    def compute_loss(self, views: List[FusionOutput], **kwargs) -> ObjectiveOutput:
        self.validate_views(views)
        
        # In a cross-modal scenario, view 0 might be Face representations, view 1 Document representations
        embeds_a = views[0].identity_embedding
        embeds_b = views[1].identity_embedding
        
        embeds_a = F.normalize(embeds_a, dim=-1)
        embeds_b = F.normalize(embeds_b, dim=-1)
        
        logit_scale = self.logit_scale.exp()
        
        logits_per_a = logit_scale * embeds_a @ embeds_b.t()
        logits_per_b = logits_per_a.t()
        
        batch_size = embeds_a.size(0)
        labels = torch.arange(batch_size, dtype=torch.long, device=embeds_a.device)
        
        loss_a = F.cross_entropy(logits_per_a, labels)
        loss_b = F.cross_entropy(logits_per_b, labels)
        
        loss = (loss_a + loss_b) / 2
        
        metrics = {
            "temperature": logit_scale.item()
        }
        
        return ObjectiveOutput(loss=loss, metrics=metrics)

    @property
    def metadata(self) -> ObjectiveMetadata:
        return ObjectiveMetadata(
            name="clip",
            family="cross_modal_contrastive",
            requires_negatives=True,
            requires_target_network=False,
            supports_multimodal=True,
            minimum_views=2
        )
