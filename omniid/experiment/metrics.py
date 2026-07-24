import os
import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class MetricsRecorder(ABC):
    @abstractmethod
    def log(self, name: str, value: float, step: Optional[int] = None):
        pass
        
    @abstractmethod
    def save(self):
        pass

class JSONMetricsRecorder(MetricsRecorder):
    """
    Default metrics recorder logging values to a JSON file.
    """
    def __init__(self, run_dir: str):
        self.metrics_path = os.path.join(run_dir, "metrics.json")
        self.history: Dict[str, Any] = {}

    def log(self, name: str, value: float, step: Optional[int] = None):
        if name not in self.history:
            self.history[name] = []
        
        record = {"value": value}
        if step is not None:
            record["step"] = step
            
        self.history[name].append(record)

    def save(self):
        with open(self.metrics_path, "w") as f:
            json.dump(self.history, f, indent=2)
