import time
import torch
from omniid.models.builder import build_encoder

class BenchmarkHarness:
    """
    Independent profiling tool to validate FEF backbones.
    """
    def __init__(self, encoder_name: str, **kwargs):
        self.encoder_name = encoder_name
        self.encoder = build_encoder(encoder_name, **kwargs)
        self.encoder.eval()

    def run(self, batch_size: int = 1):
        res = self.encoder.input_resolution
        dummy_input = torch.randn(batch_size, 3, res, res)
        
        # Profile Inference Latency
        start_time = time.time()
        with torch.no_grad():
            embeds = self.encoder.encode(dummy_input, mode="cls")
        latency = time.time() - start_time
        
        # Collect Metadata
        num_params = sum(p.numel() for p in self.encoder.parameters())
        
        return {
            "encoder": self.encoder_name,
            "latency_sec": latency,
            "output_shape": list(embeds.shape),
            "expected_dim": self.encoder.embedding_dim,
            "parameters": num_params,
            "patch_size": self.encoder.patch_size,
            "input_resolution": res,
            "preprocessing": self.encoder.get_preprocessing_transforms()
        }
