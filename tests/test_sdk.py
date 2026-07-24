import pytest
from omniid.sdk.client import DatasetClient
import os
import json

def test_sdk_integration():
    client = DatasetClient()
    manifest_client = (
        client
        .ingest("tests/mock_dataset")
        .validate()
        .assess_quality()
        .normalize(output_dir="tests/artifacts/normalized")
        .fingerprint()
        .generate_manifest(output_dir="tests/artifacts")
    )
    
    # id1 should be accepted, id2 rejected for resolution, id3 rejected for corruption
    assert manifest_client._fingerprint is not None
    assert os.path.exists("tests/artifacts/manifest.json")
    
    with open("tests/artifacts/manifest.json", "r") as f:
        manifest = json.load(f)
        
    assert manifest["statistics"]["total_samples"] == 1 # Only id1 makes it through everything
    
    # Assert determinism
    client2 = DatasetClient()
    client2.ingest("tests/mock_dataset").validate().assess_quality().normalize(output_dir="tests/artifacts/normalized").fingerprint()
    assert client2._fingerprint == manifest_client._fingerprint

def test_fingerprint_determinism():
    from omniid.data_engine.integrity.fingerprint import DatasetFingerprinter
    data1 = [{"id": "b", "meta": 1}, {"id": "a", "meta": 2}]
    data2 = [{"id": "a", "meta": 2}, {"id": "b", "meta": 1}]
    
    f1 = DatasetFingerprinter().compute(data1, {})
    f2 = DatasetFingerprinter().compute(data2, {})
    assert f1 == f2
