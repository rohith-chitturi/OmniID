from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import datetime

class InferenceRequest(BaseModel):
    """
    Standardized payload for an inference invocation.
    """
    request_id: str
    # E.g., {"image": [[224, 224, 3], ...], "layout": [...]} 
    # In a real system, these might be base64 strings or S3 URIs
    inputs: Dict[str, Any]
    modalities: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    batch_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.datetime.utcnow().isoformat())

class InferenceResponse(BaseModel):
    """
    Standardized result from the InferencePipeline.
    """
    model_config = {"protected_namespaces": ()}
    
    request_id: str
    # Assuming list of floats representing the embedding
    embedding: List[float]
    confidence: Optional[float] = None
    latency_ms: float
    active_modalities: List[str]
    model_version: str
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
