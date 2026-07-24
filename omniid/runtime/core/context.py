from enum import Enum
from dataclasses import dataclass
from typing import Any

class TrainingState(str, Enum):
    INITIALIZED = "INITIALIZED"
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    TESTING = "TESTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

@dataclass(frozen=True)
class RuntimeContext:
    """
    Immutable execution context carrying everything a run needs.
    Passed universally down to Trainers and Callbacks.
    """
    experiment_id: str
    run_id: str
    config_fingerprint: str
    metrics_recorder: Any
    checkpoint_manager: Any
    device_manager: Any
    precision_manager: Any
    seed: int
