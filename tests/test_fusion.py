import pytest
import torch
from omniid.fusion.types import ModalityEmbedding
from omniid.fusion.alignment import EmbeddingAlignmentLayer
from omniid.fusion.builder import build_fusion

def test_alignment_projection_and_masking():
    modality_dims = {"face": 100, "document": 200}
    layer = EmbeddingAlignmentLayer(target_dim=50, modality_dims=modality_dims)
    
    modalities = {
        "face": ModalityEmbedding(name="face", modality_type="img", embedding=torch.randn(2, 100), is_present=True),
        "document": ModalityEmbedding(name="document", modality_type="doc", embedding=torch.randn(2, 200), is_present=False)
    }
    
    aligned = layer(modalities, missing_strategy="mask")
    
    # Face should be projected to 50
    assert list(aligned["face"].shape) == [2, 50]
    # Document should be zeroed out
    assert torch.all(aligned["document"] == 0)
    assert list(aligned["document"].shape) == [2, 50]

def test_concat_fusion():
    modality_dims = {"face": 100, "document": 100}
    fusion = build_fusion("concat", target_dim=100, modality_dims=modality_dims)
    
    modalities = {
        "face": ModalityEmbedding(name="face", modality_type="img", embedding=torch.randn(2, 100)),
        "document": ModalityEmbedding(name="document", modality_type="doc", embedding=torch.randn(2, 100))
    }
    
    output = fusion.fuse(modalities)
    
    # Output should be 200
    assert list(output.identity_embedding.shape) == [2, 200]

def test_cross_attention_fusion():
    modality_dims = {"face": 100, "document": 100}
    fusion = build_fusion("cross_attention", target_dim=100, modality_dims=modality_dims, query_modality="face")
    
    modalities = {
        "face": ModalityEmbedding(name="face", modality_type="img", embedding=torch.randn(2, 100)),
        "document": ModalityEmbedding(name="document", modality_type="doc", embedding=torch.randn(2, 100))
    }
    
    output = fusion.fuse(modalities)
    
    # Target dim 100
    assert list(output.identity_embedding.shape) == [2, 100]
    assert output.attention_maps is not None

def test_transformer_fusion():
    modality_dims = {"face": 100, "document": 100}
    fusion = build_fusion("transformer", target_dim=100, modality_dims=modality_dims)
    
    modalities = {
        "face": ModalityEmbedding(name="face", modality_type="img", embedding=torch.randn(2, 100)),
        "document": ModalityEmbedding(name="document", modality_type="doc", embedding=torch.randn(2, 100))
    }
    
    output = fusion.fuse(modalities)
    
    # Cls token projection
    assert list(output.identity_embedding.shape) == [2, 100]
