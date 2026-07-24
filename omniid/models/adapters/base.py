from abc import ABC, abstractmethod
import torch.nn as nn
import torch

class BaseAdapter(nn.Module, ABC):
    """
    Placeholder for future Parameter-Efficient Fine-Tuning (PEFT) extensions.
    e.g., LoRA, Adapters, Prompt Tuning.
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def attach(self, encoder: nn.Module):
        """Inject parameters into the frozen encoder."""
        pass

    @abstractmethod
    def detach(self, encoder: nn.Module):
        """Remove parameters from the encoder."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
