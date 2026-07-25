import torch
import time
from typing import Dict, Any, Optional
from omniid.eval.base import BaseEvaluator
from omniid.eval.registry import EVALUATOR_REGISTRY
from omniid.eval.types import EvaluationMetadata, EvaluationReport
import numpy as np

@EVALUATOR_REGISTRY.register("verification")
class VerificationEvaluator(BaseEvaluator):
    """
    Computes standard verification metrics: EER, ROC AUC, TAR@FAR, FRR.
    """
    def __init__(self, far_targets=[1e-2, 1e-3, 1e-4]):
        self.far_targets = far_targets

    def evaluate(self, embeddings: torch.Tensor, labels: torch.Tensor, dataset_info: Optional[Dict[str, Any]] = None, **kwargs) -> EvaluationReport:
        self.validate_inputs(embeddings, labels)
        
        start_time = time.time()
        
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        # O(N^2) similarity for all pairs
        sim_matrix = torch.mm(embeddings, embeddings.t())
        
        # Generate positive and negative pairs masks
        label_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1))
        
        # Exclude self-matches
        mask = ~torch.eye(labels.size(0), dtype=torch.bool, device=embeddings.device)
        
        pos_mask = label_matrix & mask
        neg_mask = (~label_matrix) & mask
        
        pos_scores = sim_matrix[pos_mask].cpu().numpy()
        neg_scores = sim_matrix[neg_mask].cpu().numpy()
        
        metrics = {}
        artifacts = {}
        
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            # We mock the exact EER/AUC calculation for simplicity
            # In a real library, this would use sklearn.metrics or similar
            
            thresholds = np.linspace(-1, 1, 1000)
            tars = []
            fars = []
            
            for t in thresholds:
                tar = np.mean(pos_scores >= t)
                far = np.mean(neg_scores >= t)
                tars.append(tar)
                fars.append(far)
                
            tars = np.array(tars)
            fars = np.array(fars)
            
            # Find EER (where FAR == FRR, and FRR = 1 - TAR)
            frrs = 1 - tars
            eer_idx = np.nanargmin(np.abs(fars - frrs))
            eer = (fars[eer_idx] + frrs[eer_idx]) / 2
            metrics["EER"] = float(eer)
            
            # Simple AUC approximation
            auc = np.trapz(tars[::-1], fars[::-1])
            metrics["ROC_AUC"] = float(auc)
            
            # TAR @ FAR
            for target_far in self.far_targets:
                idx = np.where(fars <= target_far)[0]
                if len(idx) > 0:
                    best_idx = idx[0] # first threshold where far <= target
                    metrics[f"TAR@FAR={target_far}"] = float(tars[best_idx])
                else:
                    metrics[f"TAR@FAR={target_far}"] = 0.0
                    
            artifacts["roc_curve"] = {"fpr": fars.tolist()[::10], "tpr": tars.tolist()[::10]} # Subsampled
        else:
            metrics = {"EER": 0.0, "ROC_AUC": 0.0}
            
        runtime_ms = (time.time() - start_time) * 1000
        
        return self._create_report(metrics=metrics, runtime_ms=runtime_ms, dataset_info=dataset_info, artifacts=artifacts)

    @property
    def metadata(self) -> EvaluationMetadata:
        return EvaluationMetadata(
            name="verification",
            task_type="binary_classification",
            required_inputs=["embeddings", "labels"],
            primary_metric="EER",
            supports_multimodal=True,
            supports_batching=True
        )
