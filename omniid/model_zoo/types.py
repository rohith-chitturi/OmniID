from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import datetime

class EvaluationSummary(BaseModel):
    primary_metric: str
    summary: Dict[str, float]
    report_uri: Optional[str] = None

class ModelManifest(BaseModel):
    """
    The canonical description of a published model checkpoint.
    """
    model_config = {"protected_namespaces": ()}
    
    # Identity
    model_name: str
    version: str
    description: str

    # Framework
    framework_version: str
    python_version: str

    # Architecture
    encoder: str
    fusion: Optional[str] = None
    objective: Optional[str] = None

    # Training Provenance
    dataset_fingerprint: str
    config_fingerprint: str
    experiment_id: str

    # Checkpoint Information
    checkpoint_hash: str
    checkpoint_format: str
    checkpoint_size_mb: float

    # Evaluation
    evaluation: Optional[EvaluationSummary] = None

    # License
    license: str
    authors: List[str]
    created_at: str = datetime.datetime.utcnow().isoformat()

class ModelCard(BaseModel):
    """
    Human-readable model card containing documentation and usage constraints.
    """
    manifest: ModelManifest
    intended_use: str
    limitations: str
    datasets_used: List[str]
    supported_modalities: List[str]
    # In practice, this would also be associated with a README.md
