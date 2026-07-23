import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseFoundationEncoder(nn.Module, ABC):
    """
    Abstract Base Class for all modality-specific backbones within the OmniID Foundation Model.
    Enforces a strict interface to guarantee compatibility with multimodal fusion modules,
    adapters (PEFT), and representation heads.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract dense features from the input tensor.
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Return the final output dimension of the encoder.
        """
        pass
