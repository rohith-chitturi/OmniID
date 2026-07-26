import pytest
import os
import json
from omniid.model_zoo.types import ModelManifest, EvaluationSummary
from omniid.model_zoo.compatibility import CompatibilityValidator
from omniid.model_zoo.resolver import LocalProvider, CheckpointResolver
from omniid.model_zoo.api import ModelZoo

@pytest.fixture
def mock_manifest():
    return ModelManifest(
        model_name="omniid-dinov2-base",
        version="v1.0",
        description="Base DINOv2 encoder",
        framework_version="0.1.0",
        python_version="3.11",
        encoder="dinov2",
        dataset_fingerprint="sha256-abc123",
        config_fingerprint="sha256-def456",
        experiment_id="exp-001",
        checkpoint_hash="abcdef123456",
        checkpoint_format="pt",
        checkpoint_size_mb=340.5,
        evaluation=EvaluationSummary(
            primary_metric="mAP",
            summary={"mAP": 0.85, "Recall@1": 0.92}
        ),
        license="MIT",
        authors=["OmniID Team"]
    )

def test_compatibility_validator(mock_manifest):
    validator = CompatibilityValidator()
    
    # 1. Exact Match
    valid_config = {"encoder": "dinov2"}
    is_compat, msg = validator.validate(mock_manifest, valid_config)
    assert is_compat is True
    
    # 2. Encoder Mismatch
    invalid_config = {"encoder": "layoutlmv3"}
    is_compat, msg = validator.validate(mock_manifest, invalid_config)
    assert is_compat is False
    assert "Encoder mismatch" in msg

def test_local_provider(tmp_path, mock_manifest):
    # Setup mock zoo directory
    zoo_dir = tmp_path / "zoo"
    zoo_dir.mkdir()
    
    provider = LocalProvider(zoo_dir=str(zoo_dir))
    
    # Write mock manifest
    manifest_file = zoo_dir / "omniid-dinov2-base.json"
    manifest_file.write_text(mock_manifest.model_dump_json())
    
    # Write mock checkpoint
    ckpt_file = zoo_dir / "abcdef123456.pth"
    ckpt_file.write_text("dummy weights")
    
    # Resolve
    resolved_manifest = provider.resolve_manifest("omniid-dinov2-base")
    assert resolved_manifest["model_name"] == "omniid-dinov2-base"
    
    resolved_ckpt = provider.fetch_checkpoint("abcdef123456")
    assert os.path.exists(resolved_ckpt)

def test_model_zoo_api(tmp_path, mock_manifest, monkeypatch):
    # Patch the LocalProvider in the CheckpointResolver
    zoo_dir = tmp_path / "zoo"
    zoo_dir.mkdir()
    
    manifest_file = zoo_dir / "test-model.json"
    mock_dict = mock_manifest.model_dump()
    mock_dict["model_name"] = "test-model"
    manifest_file.write_text(json.dumps(mock_dict))
    
    ckpt_file = zoo_dir / "abcdef123456.pth"
    ckpt_file.write_text("dummy weights")
    
    zoo = ModelZoo()
    zoo.resolver.register_provider("local", LocalProvider(zoo_dir=str(zoo_dir)))
    
    # Test Info / Card
    card = zoo.get_model_card("test-model")
    assert card.manifest.model_name == "test-model"
    assert card.manifest.evaluation.primary_metric == "mAP"
    
    # Test Load (Success)
    valid_config = {"encoder": "dinov2"}
    path = zoo.load("test-model", valid_config)
    assert "abcdef123456.pth" in path
    
    # Test Load (Failure due to compatibility)
    invalid_config = {"encoder": "vit"}
    with pytest.raises(RuntimeError, match="Compatibility Validation Failed"):
        zoo.load("test-model", invalid_config)
