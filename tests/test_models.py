import pytest
import torch
from omniid.models.registry import BACKBONE_REGISTRY
from omniid.models.builder import build_encoder
from omniid.models.backbones.dinov2 import DINOv2Encoder

def test_encoder_registry_resolution():
    encoder = build_encoder("dinov2", size="vits14")
    assert isinstance(encoder, DINOv2Encoder)
    assert encoder.embedding_dim == 384
    assert encoder.patch_size == 14
    assert encoder.input_resolution == 224

def test_encoder_modes():
    encoder = build_encoder("dinov2", size="vits14")
    encoder.eval()
    
    # Mock Image Tensor [B, C, H, W]
    dummy = torch.randn(2, 3, 224, 224)
    
    cls_emb = encoder.encode(dummy, mode="cls")
    assert list(cls_emb.shape) == [2, 384]
    
    patch_emb = encoder.encode(dummy, mode="patch")
    # patches = (224/14)^2 = 16^2 = 256
    assert list(patch_emb.shape) == [2, 256, 384]

def test_frozen_toggle():
    encoder = build_encoder("dinov2", size="vits14")
    
    # By default, mock_model parameters require grad
    assert any(p.requires_grad for p in encoder.parameters())
    
    encoder.trainable = False
    assert not encoder.trainable
    assert all(not p.requires_grad for p in encoder.parameters())
    
    encoder.trainable = True
    assert encoder.trainable
    assert all(p.requires_grad for p in encoder.parameters())

def test_preprocessing_contract():
    encoder = build_encoder("dinov2")
    preproc = encoder.get_preprocessing_transforms()
    assert "mean" in preproc
    assert "std" in preproc
    assert preproc["resolution"] == 224
