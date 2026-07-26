from abc import ABC, abstractmethod
from typing import List, Any
from omniid.runtime.types import InferenceRequest

class BatchPolicy(ABC):
    """
    Defines the criteria for when a batch is ready to be executed.
    """
    @abstractmethod
    def is_ready(self, queue: List[InferenceRequest]) -> bool:
        pass

class FixedBatchPolicy(BatchPolicy):
    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size
        
    def is_ready(self, queue: List[InferenceRequest]) -> bool:
        return len(queue) >= self.batch_size

class DynamicBatchPolicy(BatchPolicy):
    def __init__(self, max_batch_size: int = 32, max_wait_ms: float = 50.0):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        # In a real async implementation, this would check timestamps
        
    def is_ready(self, queue: List[InferenceRequest]) -> bool:
        if len(queue) >= self.max_batch_size:
            return True
        # Mocking wait time logic for simplicity
        return False
