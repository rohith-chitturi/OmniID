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
    def attach(self, encoder):
        """Inject parameters into the frozen encoder."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
