from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch

@dataclass
class DocumentOutput:
    """
    Standardized representation for multi-modal Document Intelligence outputs.
    Generalizes across LayoutLM (discriminative) and Donut (generative).
    """
    document_embedding: Optional[torch.Tensor] = None
    page_embeddings: Optional[torch.Tensor] = None
    token_embeddings: Optional[torch.Tensor] = None
    layout_boxes: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    text: Optional[str] = None
    confidence_scores: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None
