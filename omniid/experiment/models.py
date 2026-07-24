from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ExperimentState(str, Enum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"

class RunArtifactManifest(BaseModel):
    config: str = "config.yaml"
    canonical_config: str = "configuration.json"
    config_fingerprint: str = "config_fingerprint.txt"
    dataset_manifest: str = "dataset_manifest.json"
    metrics: str = "metrics.json"
    logs: str = "logs.json"
    environment: str = "environment.json"

class RunMetadata(BaseModel):
    run_id: str
    experiment_id: str
    state: ExperimentState
    tags: List[str]
    seed: int
    artifacts: RunArtifactManifest

class EnvironmentSnapshot(BaseModel):
    os: str
    platform: str
    python_version: str
    pytorch_version: Optional[str] = None
    cuda_available: bool = False
    git_commit: Optional[str] = None
