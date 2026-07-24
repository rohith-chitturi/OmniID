import time
import torch
from omniid.models.builder import build_encoder

class BenchmarkHarness:
    """
    Independent profiling tool to validate FEF backbones.
    """
    def __init__(self):
        pass

    def run(self, encoder_name: str, batch_size: int = 1, **kwargs):
        load_start = time.time()
        encoder = build_encoder(encoder_name, **kwargs)
        encoder.eval()
        load_time = time.time() - load_start
        
        preproc_start = time.time()
        spec = encoder.preprocess
        preproc_time = time.time() - preproc_start

        res = spec.resolution
        dummy_input = torch.randn(batch_size, 3, res, res)
        
        # Profile Inference Latency
        inf_start = time.time()
        with torch.no_grad():
            embeds = encoder.encode(dummy_input, mode="cls")
        latency = time.time() - inf_start
        
        meta = encoder.metadata
        
        return {
            "Model": meta.name,
            "Architecture": meta.architecture,
            "Embedding": meta.embedding_dim,
            "Parameters": meta.parameter_count,
            "Load Time": f"{load_time:.4f}s",
            "Preprocessing Time": f"{preproc_time:.4f}s",
            "Inference": f"{latency:.4f}s",
            "Output Shape": list(embeds.shape),
            "Memory": "N/A (CPU fallback mock)" # Mocking memory tracking for now
        }
