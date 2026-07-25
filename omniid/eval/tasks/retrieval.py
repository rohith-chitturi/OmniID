import torch
import time
from typing import Dict, Any, Optional
from omniid.eval.base import BaseEvaluator
from omniid.eval.registry import EVALUATOR_REGISTRY
from omniid.eval.types import EvaluationMetadata, EvaluationReport

@EVALUATOR_REGISTRY.register("retrieval")
class RetrievalEvaluator(BaseEvaluator):
    """
    Computes standard retrieval metrics: mAP, Recall@K, Precision@K, MRR.
    """
    def __init__(self, k_values=[1, 5, 10]):
        self.k_values = k_values

    def evaluate(self, embeddings: torch.Tensor, labels: torch.Tensor, dataset_info: Optional[Dict[str, Any]] = None, **kwargs) -> EvaluationReport:
        self.validate_inputs(embeddings, labels)
        
        start_time = time.time()
        
        # Normalize embeddings for cosine similarity
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        # Compute all-to-all similarity matrix (naive O(N^2) for demonstration)
        sim_matrix = torch.mm(embeddings, embeddings.t())
        
        # Mask out self-similarity
        mask = torch.eye(labels.size(0), dtype=torch.bool, device=embeddings.device)
        sim_matrix.masked_fill_(mask, -1.0)
        
        # Sort similarities to get ranking
        sorted_indices = torch.argsort(sim_matrix, dim=-1, descending=True)
        sorted_labels = labels[sorted_indices]
        
        # Match matrix: 1 if matches query label, 0 otherwise
        matches = (sorted_labels == labels.unsqueeze(1)).float()
        
        metrics = {}
        
        # Compute Recall@K and Precision@K
        for k in self.k_values:
            matches_k = matches[:, :k]
            # Has at least one relevant item in top K
            recall_k = (matches_k.sum(dim=1) > 0).float().mean().item()
            precision_k = (matches_k.sum(dim=1) / k).mean().item()
            metrics[f"Recall@{k}"] = recall_k
            metrics[f"Precision@{k}"] = precision_k
            
        # Mean Reciprocal Rank (MRR)
        # Find first non-zero match index for each query
        first_match_idx = (matches == 1).long().argmax(dim=1)
        has_match = (matches == 1).sum(dim=1) > 0
        mrr = (1.0 / (first_match_idx[has_match] + 1).float()).mean().item() if has_match.any() else 0.0
        metrics["MRR"] = mrr
        
        # Mean Average Precision (mAP)
        # Simplified calculation for demonstration
        cum_matches = matches.cumsum(dim=1)
        positions = torch.arange(1, matches.size(1) + 1, device=matches.device).float()
        precision_at_i = cum_matches / positions
        ap = (precision_at_i * matches).sum(dim=1) / (matches.sum(dim=1).clamp(min=1))
        mAP = ap.mean().item()
        metrics["mAP"] = mAP
        
        runtime_ms = (time.time() - start_time) * 1000
        
        return self._create_report(metrics=metrics, runtime_ms=runtime_ms, dataset_info=dataset_info)

    @property
    def metadata(self) -> EvaluationMetadata:
        return EvaluationMetadata(
            name="retrieval",
            task_type="ranking",
            required_inputs=["embeddings", "labels"],
            primary_metric="mAP",
            supports_multimodal=True,
            supports_batching=True
        )
