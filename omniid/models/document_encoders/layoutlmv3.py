import torch
import torch.nn as nn
from typing import Dict, Any
from omniid.models.document_base import BaseDocumentEncoder
from omniid.models.registry import BACKBONE_REGISTRY
from omniid.models.weights import WeightManager
from omniid.models.metadata import ModelMetadata
from omniid.models.preprocess import DocumentPreprocessingSpec
from omniid.models.outputs import DocumentOutput

@BACKBONE_REGISTRY.register("layoutlmv3")
class LayoutLMv3Encoder(BaseDocumentEncoder):
    def __init__(self, size: str = "base"):
        super().__init__()
        self.size = size
        self.mock_model = nn.Linear(3 * 224 * 224, 768)

    def load_weights(self):
        manager = WeightManager()
        path = manager.load_weight(f"https://huggingface.co/microsoft/layoutlmv3-{self.size}/pytorch_model.bin", expected_sha256="mock_hash")
        return path

    def encode(self, document_input: Dict[str, Any], **kwargs) -> DocumentOutput:
        # Mock forward pass
        img_tensor = torch.randn(1, 3, 224, 224) # Mocks pipeline['image_tensor']
        B = img_tensor.shape[0]
        flat = img_tensor.view(B, -1)
        embeds = self.mock_model(flat) # [B, 768]
        
        # Mock sequences
        seq_embeds = embeds.unsqueeze(1).repeat(1, 512, 1) # [B, 512, 768]
        
        # LayoutLMv3 produces both a global representation and sequence tokens
        return DocumentOutput(
            document_embedding=embeds,
            token_embeddings=seq_embeds,
            layout_boxes=torch.tensor(document_input.get("layout_boxes", [])),
            attention_mask=torch.ones(1, 512)
        )

    @property
    def preprocess(self) -> DocumentPreprocessingSpec:
        return DocumentPreprocessingSpec(
            resolution=224,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            bounding_box_scale=1000,
            requires_ocr=True,
            max_pages=1
        )

    @property
    def metadata(self) -> ModelMetadata:
        num_params = sum(p.numel() for p in self.parameters())
        return ModelMetadata(
            name=f"layoutlmv3_{self.size}",
            architecture="Transformer with Spatial Embeddings",
            input_resolution=224,
            patch_size=16,
            embedding_dim=768,
            parameter_count=num_params,
            pretrained_source="Microsoft",
            license="MIT",
            supports_layout=True,
            supports_ocr=True,
            max_sequence_length=512
        )
