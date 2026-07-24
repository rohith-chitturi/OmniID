import torch
import torch.nn as nn
from omniid.models.base import BaseFoundationEncoder
from omniid.models.registry import BACKBONE_REGISTRY
from omniid.models.weights import WeightManager
from omniid.models.metadata import ModelMetadata
from omniid.models.preprocess import PreprocessingSpec

@BACKBONE_REGISTRY.register("dinov2")
class DINOv2Encoder(BaseFoundationEncoder):
    def __init__(self, size: str = "vits14"):
        super().__init__()
        self.size = size
        self.mock_model = nn.Linear(3 * 224 * 224, self.embedding_dim)

    def load_weights(self):
        manager = WeightManager()
        # Using a mock expected sha256
        path = manager.load_weight(f"https://dl.fbaipublicfiles.com/dinov2/dinov2_{self.size}.pth", expected_sha256="mock_hash")
        return path

    def encode(self, image: torch.Tensor, mode: str = "cls") -> torch.Tensor:
        if mode not in ["cls", "patch", "pooled"]:
            raise ValueError(f"Invalid extraction mode: {mode}")
            
        B = image.shape[0]
        flat = image.view(B, -1)
        embeds = self.mock_model(flat)
        
        if mode == "cls" or mode == "pooled":
            return embeds
        elif mode == "patch":
            num_patches = (224 // 14) ** 2
            return embeds.unsqueeze(1).repeat(1, num_patches, 1)

    @property
    def preprocess(self) -> PreprocessingSpec:
        return PreprocessingSpec(
            resolution=224,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            interpolation="bicubic",
            color_space="RGB"
        )

    @property
    def embedding_dim(self) -> int:
        dims = {"vits14": 384, "vitb14": 768, "vitl14": 1024, "vitg14": 1536}
        return dims.get(self.size, 384)

    @property
    def metadata(self) -> ModelMetadata:
        num_params = sum(p.numel() for p in self.parameters())
        return ModelMetadata(
            name=f"dinov2_{self.size}",
            architecture="Vision Transformer",
            input_resolution=224,
            patch_size=14,
            embedding_dim=self.embedding_dim,
            parameter_count=num_params,
            pretrained_source="Meta AI",
            license="Apache 2.0"
        )
