from abc import ABC, abstractmethod
from typing import Dict, Any
import torch.nn as nn
import torch

class BaseFoundationEncoder(nn.Module, ABC):
    """
    Universal contract for all OmniID Vision Backbones.
    """
    def __init__(self):
        super().__init__()
        self._trainable = False

    @abstractmethod
    def encode(self, image: torch.Tensor, mode: str = "cls") -> torch.Tensor:
        """
        Extract features. Mode can be 'cls', 'patch', or 'pooled'.
        """
        pass

    @abstractmethod
    def get_preprocessing_transforms(self) -> Dict[str, Any]:
        """
        Returns the expected mean, std, and input resolution for this specific encoder.
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def patch_size(self) -> int:
        pass

    @property
    @abstractmethod
    def input_resolution(self) -> int:
        pass
        
    @property
    def trainable(self) -> bool:
        return self._trainable
        
    @trainable.setter
    def trainable(self, value: bool):
        self._trainable = value
        for param in self.parameters():
            param.requires_grad = value
