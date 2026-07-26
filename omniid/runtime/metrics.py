from pydantic import BaseModel
from typing import Dict
import time

class RuntimeMetrics(BaseModel):
    request_latency_ms: float
    throughput_req_sec: float
    queue_depth: int

class ResourceMetrics(BaseModel):
    gpu_memory_mb: float
    cpu_utilization_pct: float
    batch_utilization_pct: float

class ModelMetrics(BaseModel):
    load_time_ms: float
    inference_time_ms: float
    preprocessing_time_ms: float

class MetricsCollector:
    """
    Centralized collector for operational telemetry.
    """
    def __init__(self):
        self._latencies = []
        self._load_times = []
        
    def record_latency(self, latency_ms: float):
        self._latencies.append(latency_ms)
        
    def record_load_time(self, load_time_ms: float):
        self._load_times.append(load_time_ms)
        
    def get_summary(self) -> Dict[str, float]:
        avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
        avg_load = sum(self._load_times) / len(self._load_times) if self._load_times else 0.0
        return {
            "avg_latency_ms": avg_latency,
            "avg_load_time_ms": avg_load,
            "total_requests": len(self._latencies)
        }
