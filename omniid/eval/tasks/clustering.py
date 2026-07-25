import torch
import time
from typing import Dict, Any, Optional
from omniid.eval.base import BaseEvaluator
from omniid.eval.registry import EVALUATOR_REGISTRY
from omniid.eval.types import EvaluationMetadata, EvaluationReport
import numpy as np
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

@EVALUATOR_REGISTRY.register("clustering")
class ClusteringEvaluator(BaseEvaluator):
    """
    Computes standard clustering metrics: NMI, ARI.
    Requires scikit-learn for KMeans clustering.
    """
    def __init__(self, n_clusters: Optional[int] = None):
        self.n_clusters = n_clusters

    def evaluate(self, embeddings: torch.Tensor, labels: torch.Tensor, dataset_info: Optional[Dict[str, Any]] = None, **kwargs) -> EvaluationReport:
        self.validate_inputs(embeddings, labels)
        
        start_time = time.time()
        metrics = {}
        
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for ClusteringEvaluator. Run 'pip install scikit-learn'")
            
        embeddings_np = embeddings.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # If n_clusters not provided, use the number of unique ground truth labels
        n_clusters = self.n_clusters if self.n_clusters is not None else len(np.unique(labels_np))
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        pred_labels = kmeans.fit_predict(embeddings_np)
        
        # Compute metrics
        nmi = normalized_mutual_info_score(labels_np, pred_labels)
        ari = adjusted_rand_score(labels_np, pred_labels)
        
        metrics["NMI"] = float(nmi)
        metrics["ARI"] = float(ari)
        
        runtime_ms = (time.time() - start_time) * 1000
        
        return self._create_report(metrics=metrics, runtime_ms=runtime_ms, dataset_info=dataset_info)

    @property
    def metadata(self) -> EvaluationMetadata:
        return EvaluationMetadata(
            name="clustering",
            task_type="unsupervised_grouping",
            required_inputs=["embeddings", "labels"],
            primary_metric="NMI",
            supports_multimodal=True,
            supports_batching=False # Clustering typically requires the full dataset at once
        )
