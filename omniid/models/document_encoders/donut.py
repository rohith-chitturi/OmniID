import torch
import torch.nn as nn
from typing import Dict, Any
from omniid.models.document_base import BaseDocumentEncoder
from omniid.models.registry import BACKBONE_REGISTRY
from omniid.models.weights import WeightManager
from omniid.models.metadata import ModelMetadata
from omniid.models.preprocess import DocumentPreprocessingSpec
from omniid.models.outputs import DocumentOutput

@BACKBONE_REGISTRY.register("donut")
class DonutEncoder(BaseDocumentEncoder):
    """
    OCR-free generative document understanding framework.
    """
    def __init__(self, size: str = "base"):
        super().__init__()
        self.size = size
        self.mock_model = nn.Linear(3 * 224 * 224, 1024)

    def load_weights(self):
        manager = WeightManager()
        path = manager.load_weight(f"https://huggingface.co/naver-clova-ix/donut-{self.size}/pytorch_model.bin", expected_sha256="mock_hash")
        return path

    def encode(self, document_input: Dict[str, Any], **kwargs) -> DocumentOutput:
        # Mock forward pass
        img_tensor = torch.randn(1, 3, 224, 224) 
        B = img_tensor.shape[0]
        flat = img_tensor.view(B, -1)
        embeds = self.mock_model(flat) # [B, 1024]
        
        # Donut is generative, so it outputs text sequences instead of token embeddings natively
        return DocumentOutput(
            document_embedding=embeds,
            text="<s_receipt><s_company>OmniCorp</s_company></s_receipt>"
        )

    @property
    def preprocess(self) -> DocumentPreprocessingSpec:
        return DocumentPreprocessingSpec(
            resolution=224,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            bounding_box_scale=None, # Donut is OCR-free
            requires_ocr=False,
            max_pages=1
        )

    @property
    def metadata(self) -> ModelMetadata:
        num_params = sum(p.numel() for p in self.parameters())
        return ModelMetadata(
            name=f"donut_{self.size}",
            architecture="Vision-Encoder-Decoder",
            input_resolution=224,
            patch_size=16,
            embedding_dim=1024,
            parameter_count=num_params,
            pretrained_source="Naver Clova",
            license="MIT",
            supports_layout=False,
            supports_ocr=False,
            supports_generation=True,
            max_sequence_length=1536
        )
