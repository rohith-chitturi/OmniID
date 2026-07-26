from typing import List, Callable, Any
from omniid.runtime.types import InferenceRequest, InferenceResponse
from omniid.runtime.batching import BatchPolicy, FixedBatchPolicy

class BatchScheduler:
    """
    Accumulates requests and flushes them to the execution engine based on a BatchPolicy.
    """
    def __init__(self, policy: BatchPolicy = None):
        self.policy = policy or FixedBatchPolicy(batch_size=1)
        self.queue: List[InferenceRequest] = []
        
    def submit(self, request: InferenceRequest) -> bool:
        """
        Submits a request to the queue. Returns True if the batch is ready to execute.
        """
        self.queue.append(request)
        return self.policy.is_ready(self.queue)
        
    def flush(self) -> List[InferenceRequest]:
        """
        Retrieves the current batch and clears the queue.
        """
        batch = self.queue[:]
        self.queue.clear()
        return batch
