from abc import ABC, abstractmethod
from typing import Any

class BaseExporter(ABC):
    """
    Standard contract for exporting a model to a production artifact format.
    """
    @abstractmethod
    def export(self, model: Any, output_path: str, dummy_input: Any = None) -> str:
        """
        Exports the model and returns the path to the exported artifact.
        """
        pass
