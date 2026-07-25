import pytest
import torch
from omniid.eval.builder import build_evaluator

def test_evaluator_registry():
    retrieval = build_evaluator("retrieval")
    verification = build_evaluator("verification")
    clustering = build_evaluator("clustering")
    
    assert retrieval.metadata.primary_metric == "mAP"
    assert verification.metadata.primary_metric == "EER"
    assert clustering.metadata.primary_metric == "NMI"

def test_retrieval_evaluator():
    evaluator = build_evaluator("retrieval", k_values=[1])
    
    # Perfect retrieval mock
    embeddings = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0]
    ])
    labels = torch.tensor([0, 0, 1, 1])
    
    report = evaluator.evaluate(embeddings, labels)
    assert report.metrics["mAP"] == 1.0
    assert report.metrics["Recall@1"] == 1.0

def test_verification_evaluator():
    evaluator = build_evaluator("verification")
    
    # Perfect separation mock
    embeddings = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0]
    ])
    labels = torch.tensor([0, 0, 1, 1])
    
    report = evaluator.evaluate(embeddings, labels)
    
    assert report.metrics["EER"] == 0.0
    assert report.metrics["ROC_AUC"] == 1.0
