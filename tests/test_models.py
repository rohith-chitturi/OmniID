import pytest
import torch
import os
import shutil
from omniid.models.registry import BACKBONE_REGISTRY
from omniid.models.builder import build_encoder
from omniid.models.backbones.dinov2 import DINOv2Encoder
from omniid.models.weights import WeightManager

def test_encoder_registry_resolution():
    encoder = build_encoder("dinov2", size="vits14")
    assert isinstance(encoder, DINOv2Encoder)
    assert encoder.embedding_dim == 384
    assert encoder.metadata.name == "dinov2_vits14"

def test_encoder_modes():
    encoder = build_encoder("dinov2", size="vits14")
    encoder.eval()
    dummy = torch.randn(2, 3, 224, 224)
    
    cls_emb = encoder.encode(dummy, mode="cls")
    assert list(cls_emb.shape) == [2, 384]
    
    patch_emb = encoder.encode(dummy, mode="patch")
    assert list(patch_emb.shape) == [2, 256, 384]

def test_frozen_toggle():
    encoder = build_encoder("dinov2", size="vits14")
    encoder.unfreeze()
    assert encoder.trainable
    assert all(p.requires_grad for p in encoder.parameters())
    
    encoder.freeze()
    assert not encoder.trainable
    assert all(not p.requires_grad for p in encoder.parameters())

def test_preprocessing_contract():
    encoder = build_encoder("dinov2")
    preproc = encoder.preprocess
    assert preproc.resolution == 224
    assert preproc.interpolation == "bicubic"
    assert preproc.color_space == "RGB"

def test_weight_manager():
    cache = "tests/test_weights"
    if os.path.exists(cache):
        shutil.rmtree(cache)
    manager = WeightManager(cache_dir=cache)
    
    # Mock download
    path = manager.load_weight("https://mock.com/model.pth")
    assert os.path.exists(path)
    
    # Verify checksum logic
    with open(path, "w") as f:
        f.write("test")
    # SHA256 of "test" is 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08
    manager._verify_checksum(path, "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08")
