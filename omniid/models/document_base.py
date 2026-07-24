from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Any
from omniid.models.metadata import ModelMetadata
from omniid.models.preprocess import DocumentPreprocessingSpec
from omniid.models.outputs import DocumentOutput

class BaseDocumentEncoder(nn.Module, ABC):
    """
    Universal contract for all OmniID Document Encoders (LayoutLM, Donut, etc.).
    """
    def __init__(self):
        super().__init__()
        self._trainable = False

    @abstractmethod
    def load_weights(self):
        """Invoke the WeightManager to load specific assets."""
        pass

    @abstractmethod
    def encode(self, document_input: Any, **kwargs) -> DocumentOutput:
        """
        Extract features or generate sequence.
        Expects pre-processed Document Inputs (handled by the Pipeline).
        """
        pass

    @property
    @abstractmethod
    def preprocess(self) -> DocumentPreprocessingSpec:
        """Returns the specific preprocessing requirements for this document encoder."""
        pass

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        pass
        
    def freeze(self):
        self._trainable = False
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        self._trainable = True
        for param in self.parameters():
            param.requires_grad = True
            
    @property
    def trainable(self) -> bool:
        return self._trainable
