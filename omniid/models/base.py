from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from omniid.models.metadata import ModelMetadata
from omniid.models.preprocess import PreprocessingSpec

class BaseFoundationEncoder(nn.Module, ABC):
    """
    Universal contract for all OmniID Vision Backbones.
    """
    def __init__(self):
        super().__init__()
        self._trainable = False

    @abstractmethod
    def load_weights(self):
        """Invoke the WeightManager to load specific assets."""
        pass

    @abstractmethod
    def encode(self, image: torch.Tensor, mode: str = "cls") -> torch.Tensor:
        """
        Extract features. Mode can be 'cls', 'patch', or 'pooled'.
        """
        pass

    @property
    @abstractmethod
    def preprocess(self) -> PreprocessingSpec:
        """Returns the specific preprocessing requirements for this encoder."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
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
