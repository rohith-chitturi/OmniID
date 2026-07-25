import pytest
import torch
from omniid.objectives.builder import build_objective
from omniid.fusion.types import FusionOutput

def create_mock_view(dim=128):
    return FusionOutput(
        identity_embedding=torch.randn(2, dim),
        modality_embeddings={},
        active_modalities=["face"]
    )

def test_registry_resolution():
    simclr = build_objective("simclr")
    vicreg = build_objective("vicreg")
    byol = build_objective("byol")
    clip = build_objective("clip")
    
    assert simclr.metadata.family == "contrastive"
    assert vicreg.metadata.family == "regularization"
    assert byol.metadata.family == "distillation"
    assert clip.metadata.family == "cross_modal_contrastive"

def test_simclr_loss():
    objective = build_objective("simclr")
    v1 = create_mock_view()
    v2 = create_mock_view()
    
    output = objective.compute_loss([v1, v2])
    assert output.loss.ndim == 0 # scalar
    assert "contrastive_accuracy" in output.metrics

def test_vicreg_loss():
    objective = build_objective("vicreg")
    v1 = create_mock_view()
    v2 = create_mock_view()
    
    output = objective.compute_loss([v1, v2])
    assert output.loss.ndim == 0 # scalar
    assert "variance" in output.auxiliary_losses

def test_missing_views():
    objective = build_objective("simclr")
    v1 = create_mock_view()
    
    with pytest.raises(ValueError):
        objective.compute_loss([v1]) # Needs 2
