from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseFaceGenerator(ABC):
    @abstractmethod
    def generate(self, persona: Dict[str, Any], seed: int, output_path: str) -> str:
        pass

class BaseDocumentGenerator(ABC):
    @abstractmethod
    def generate(self, persona: Dict[str, Any], template: str, seed: int, output_path: str) -> str:
        pass

class BaseSignatureGenerator(ABC):
    @abstractmethod
    def generate(self, name: str, seed: int, output_path: str) -> str:
        pass

class BaseVoiceGenerator(ABC):
    @abstractmethod
    def generate(self, text: str, seed: int, output_path: str) -> str:
        pass
