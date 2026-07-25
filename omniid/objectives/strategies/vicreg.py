import torch
import torch.nn.functional as F
from typing import List
from omniid.objectives.base import BaseObjective
from omniid.objectives.registry import OBJECTIVE_REGISTRY
from omniid.objectives.types import ObjectiveOutput, ObjectiveMetadata
from omniid.fusion.types import FusionOutput

@OBJECTIVE_REGISTRY.register("vicreg")
class VICRegObjective(BaseObjective):
    """
    Variance-Invariance-Covariance Regularization.
    Does not require negative samples.
    """
    def __init__(self, sim_coeff: float = 25.0, var_coeff: float = 25.0, cov_coeff: float = 1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff

    def compute_loss(self, views: List[FusionOutput], **kwargs) -> ObjectiveOutput:
        self.validate_views(views)
        
        x = views[0].identity_embedding
        y = views[1].identity_embedding
        
        # Invariance (Sim) loss
        repr_loss = F.mse_loss(x, y)
        
        # Variance loss
        std_x = torch.sqrt(x.var(dim=0) + 1e-4)
        std_y = torch.sqrt(y.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
        
        # Covariance loss
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        cov_x = (x.T @ x) / (x.size(0) - 1)
        cov_y = (y.T @ y) / (y.size(0) - 1)
        
        cov_loss = (self.off_diagonal(cov_x).pow(2).sum() / x.size(1)) + \
                   (self.off_diagonal(cov_y).pow(2).sum() / y.size(1))
                   
        loss = (self.sim_coeff * repr_loss) + (self.var_coeff * std_loss) + (self.cov_coeff * cov_loss)
        
        auxiliary_losses = {
            "invariance": repr_loss,
            "variance": std_loss,
            "covariance": cov_loss
        }
        
        return ObjectiveOutput(loss=loss, auxiliary_losses=auxiliary_losses)

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    @property
    def metadata(self) -> ObjectiveMetadata:
        return ObjectiveMetadata(
            name="vicreg",
            family="regularization",
            requires_negatives=False,
            requires_target_network=False,
            supports_multimodal=False,
            minimum_views=2
        )
