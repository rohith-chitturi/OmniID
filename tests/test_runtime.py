import pytest
import os
import json
from omniid.runtime.types import InferenceRequest, InferenceResponse
from omniid.runtime.validation import RuntimeValidator
from omniid.runtime.batching import FixedBatchPolicy
from omniid.runtime.scheduler import BatchScheduler
from omniid.runtime.exporters.registry import EXPORTER_REGISTRY
from omniid.model_zoo.types import ModelManifest
from omniid.runtime.pipeline import InferencePipeline

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
        license="MIT",
        authors=["OmniID Team"]
    )

def test_inference_request_serialization():
    req = InferenceRequest(
        request_id="req-123",
        inputs={"image": [1, 2, 3]},
        modalities=["vision"]
    )
    serialized = req.model_dump_json()
    deserialized = InferenceRequest.model_validate_json(serialized)
    assert deserialized.request_id == "req-123"

def test_runtime_validator(mock_manifest):
    validator = RuntimeValidator(mock_manifest)
    
    # Valid
    req_valid = InferenceRequest(
        request_id="1",
        inputs={"image": [1, 2]},
        modalities=["vision"]
    )
    validator.validate_request(req_valid) # Should not raise
    
    # Invalid Modality
    req_invalid_mod = InferenceRequest(
        request_id="2",
        inputs={"image": [1, 2]},
        modalities=["text"] # DINOv2 doesn't support text
    )
    with pytest.raises(ValueError, match="does not support modality"):
        validator.validate_request(req_invalid_mod)
        
    # Invalid Batch Size
    req_huge = InferenceRequest(
        request_id="3",
        inputs={"image": list(range(100))},
        modalities=["vision"]
    )
    with pytest.raises(ValueError, match="exceeds maximum allowed"):
        validator.validate_request(req_huge)

def test_batch_scheduler():
    scheduler = BatchScheduler(FixedBatchPolicy(batch_size=2))
    
    req1 = InferenceRequest(request_id="1", inputs={"a": [1]}, modalities=["vision"])
    req2 = InferenceRequest(request_id="2", inputs={"a": [2]}, modalities=["vision"])
    
    is_ready = scheduler.submit(req1)
    assert not is_ready
    
    is_ready = scheduler.submit(req2)
    assert is_ready
    
    batch = scheduler.flush()
    assert len(batch) == 2
    assert len(scheduler.queue) == 0

def test_exporter_registry(tmp_path):
    exporter = EXPORTER_REGISTRY.build("onnx")
    out_path = tmp_path / "model.onnx"
    
    # Requires dummy_input
    with pytest.raises(ValueError):
        exporter.export("dummy", str(out_path))
        
    res_path = exporter.export("dummy", str(out_path), dummy_input="tensor")
    assert os.path.exists(res_path)

def test_inference_pipeline_execution(tmp_path, monkeypatch, mock_manifest):
    # Mocking zoo for pipeline
    from omniid.model_zoo.api import ModelZoo
    from omniid.model_zoo.resolver import LocalProvider
    
    zoo_dir = tmp_path / "zoo"
    zoo_dir.mkdir()
    
    manifest = mock_manifest
    manifest_file = zoo_dir / f"{manifest.model_name}.json"
    manifest_file.write_text(manifest.model_dump_json())
    
    ckpt_file = zoo_dir / f"{manifest.checkpoint_hash}.pth"
    ckpt_file.write_text("dummy")
    
    zoo = ModelZoo()
    zoo.resolver.register_provider("local", LocalProvider(zoo_dir=str(zoo_dir)))
    
    pipeline = InferencePipeline(manifest.model_name, zoo=zoo)
    
    req = InferenceRequest(
        request_id="eval-1",
        inputs={"image": [1]},
        modalities=["vision"]
    )
    
    response = pipeline.predict(req)
    
    assert response.request_id == "eval-1"
    assert len(response.embedding) == 128
    assert response.confidence == 0.95
    assert pipeline.metrics.get_summary()["total_requests"] == 1
