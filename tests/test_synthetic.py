import pytest
from omniid.synthetic.generator import SyntheticIdentityGenerator
from omniid.sdk.client import DatasetClient
import os
import shutil

def test_synthetic_to_data_engine():
    output_dir = "tests/synthetic_dataset"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    generator = SyntheticIdentityGenerator()
    generator.generate(
        scenario="passport_verification",
        count=5,
        seed=42,
        output_dir=output_dir
    )
    
    # Verify generation
    assert os.path.exists(os.path.join(output_dir, "face"))
    assert os.path.exists(os.path.join(output_dir, "metadata.json"))
    
    # Verify integration with Data Engine SDK
    client = DatasetClient()
    manifest_client = (
        client
        .ingest(output_dir)
        .validate()
        .assess_quality()
        .normalize(output_dir="tests/artifacts/synthetic_normalized")
        .fingerprint()
        .generate_manifest(output_dir="tests/artifacts/synthetic_manifest")
    )
    
    # Ensure all generated identities were accepted
    assert len(manifest_client._validation_result.rejected) == 0
    assert len(manifest_client._quality_result.rejected) == 0
