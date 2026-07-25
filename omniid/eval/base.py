from abc import ABC, abstractmethod
import torch
from typing import Dict, Any, Optional
from omniid.eval.types import EvaluationMetadata, EvaluationReport
import time

class BaseEvaluator(ABC):
    """
    Universal contract for all Downstream Evaluation tasks.
    """
    
    @abstractmethod
    def evaluate(self, embeddings: torch.Tensor, labels: torch.Tensor, dataset_info: Optional[Dict[str, Any]] = None, **kwargs) -> EvaluationReport:
        """
        Executes the evaluation task on the provided embeddings and ground-truth labels.
        """
        pass

    @property
    @abstractmethod
    def metadata(self) -> EvaluationMetadata:
        """Returns the schema defining this evaluator's capabilities."""
        pass

    def validate_inputs(self, embeddings: torch.Tensor, labels: torch.Tensor):
        if embeddings.size(0) != labels.size(0):
            raise ValueError(f"Embeddings ({embeddings.size(0)}) and Labels ({labels.size(0)}) must have the same batch dimension.")
            
    def _create_report(self, metrics: Dict[str, float], runtime_ms: float, dataset_info: Optional[Dict[str, Any]] = None, artifacts: Optional[Dict[str, Any]] = None) -> EvaluationReport:
        return EvaluationReport(
            evaluator_name=self.metadata.name,
            dataset_info=dataset_info or {"name": "unknown", "size": 0},
            metrics=metrics,
            runtime={"total_ms": runtime_ms},
            artifacts=artifacts or {}
        )
