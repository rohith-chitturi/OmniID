import time
import torch
from typing import Dict, Any
from omniid.fusion.builder import build_fusion
from omniid.fusion.types import ModalityEmbedding

class FusionBenchmarkHarness:
    def __init__(self, strategy_name: str, target_dim: int, modality_dims: Dict[str, int], missing_strategy: str = "mask"):
        self.strategy_name = strategy_name
        self.fusion = build_fusion(
            name=strategy_name, 
            target_dim=target_dim, 
            modality_dims=modality_dims, 
            missing_strategy=missing_strategy
        )

    def run(self, batch_size: int = 2) -> Dict[str, Any]:
        self.fusion.eval()
        
        # Mock inputs
        modalities = {}
        for name, dim in self.fusion.alignment_layer.projections.items():
            modalities[name] = ModalityEmbedding(
                name=name,
                modality_type="mock",
                embedding=torch.randn(batch_size, dim),
                is_present=True
            )
            
        # Mock projection latency
        proj_start = time.time()
        with torch.no_grad():
            self.fusion.alignment_layer(modalities)
        proj_time = time.time() - proj_start
        
        # Mock fusion latency
        fuse_start = time.time()
        with torch.no_grad():
            output = self.fusion.fuse(modalities)
        fuse_time = time.time() - fuse_start
        
        num_params = sum(p.numel() for p in self.fusion.parameters())
        
        return {
            "Fusion Strategy": self.strategy_name,
            "Target Dim": self.fusion.target_dim,
            "Parameters": num_params,
            "Projection Time": f"{proj_time:.4f}s",
            "Fusion Time": f"{fuse_time:.4f}s",
            "Output Shape": list(output.identity_embedding.shape),
            "Attention Support": output.attention_maps is not None
        }
