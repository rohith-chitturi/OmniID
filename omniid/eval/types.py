from pydantic import BaseModel
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import datetime

class EvaluationMetadata(BaseModel):
    """
    Standardized schema for Evaluator properties.
    """
    name: str
    task_type: str
    required_inputs: List[str]
    primary_metric: str
    supports_multimodal: bool
    supports_batching: bool

@dataclass
class EvaluationReport:
    """
    A reproducible artifact capturing the results of an evaluation run.
    """
    evaluator_name: str
    dataset_info: Dict[str, Any]
    metrics: Dict[str, float]
    runtime: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
